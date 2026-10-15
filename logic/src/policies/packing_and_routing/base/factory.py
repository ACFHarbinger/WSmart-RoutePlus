"""Factory for joint packing-and-routing policies. Construction is ``JOINT.create``."""

from typing import Any, Optional

from logic.src.utils.registry import JOINT

from .base_joint_policy import BaseJointPolicy
from .registry import JointPolicyRegistry


class JointPolicyFactory:
    """Create a joint policy by registry name."""

    @staticmethod
    def create(name: str, config: Optional[Any] = None) -> BaseJointPolicy:
        if JointPolicyRegistry.get(name) is None:
            raise ValueError(f"Unknown joint policy: {name}")
        return JOINT.create(name, cfg=config)

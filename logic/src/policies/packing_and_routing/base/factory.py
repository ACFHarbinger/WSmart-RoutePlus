"""Factory for joint packing and routing policies.

Provides a centralized mechanism to instantiate registered policies by name.

Attributes:
    JointPolicyFactory: Factory class for instantiating policies.

Example:
    >>> from logic.src.policies.packing_and_routing.base.factory import JointPolicyFactory
    >>> policy = JointPolicyFactory.create("jgo")
"""

from typing import Any, Optional

from .base_joint_policy import BaseJointPolicy
from .registry import JointPolicyRegistry


class JointPolicyFactory:
    """Factory for creating joint packing and routing policies.

    Provides a centralized mechanism to instantiate registered policies by name.

    Attributes:
        None
    """

    @staticmethod
    def create(name: str, config: Optional[Any] = None) -> BaseJointPolicy:
        """Create a joint policy by name.

        Args:
            name (str): Unique name identifier for the policy.
            config (Optional[Any], optional): Configuration parameters for the
                policy instance. Defaults to None.

        Returns:
            BaseJointPolicy: The instantiated policy.

        Raises:
            ValueError: If no policy is registered under the given name.
        """
        policy_cls = JointPolicyRegistry.get(name)
        if policy_cls is None:
            raise ValueError(f"Unknown joint policy: {name}")
        return policy_cls(config)

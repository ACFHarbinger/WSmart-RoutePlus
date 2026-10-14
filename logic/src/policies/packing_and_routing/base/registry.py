"""Registry for joint packing and routing policies.

Maintains a mapping of policy names to their respective classes, enabling
dynamic instantiation through the factory.

Attributes:
    JointPolicyRegistry: The registry class defined in this module.

Example:
    >>> from logic.src.policies.packing_and_routing.base.registry import JointPolicyRegistry
    >>> policy_cls = JointPolicyRegistry.get("jgo")
"""

from typing import Any, Dict, Optional, Type


class JointPolicyRegistry:
    """Registry for joint packing and routing policies.

    Maintains a mapping of policy names to their respective classes, enabling
    dynamic instantiation through the factory.

    Attributes:
        _registry (Dict[str, Type]): Internal mapping of policy names to class types.
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a joint packing and routing policy class.

        Args:
            name (str): Unique name identifier for the policy.

        Returns:
            Callable: A decorator that registers the subclass.
        """

        def decorator(subclass: Type[Any]):
            """Internal decorator that adds the subclass to the registry mapping.

            Args:
                subclass (Type[Any]): The policy class to be registered.

            Returns:
                Type[Any]: The same subclass, enabling decorator chaining.
            """
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Retrieve a registered policy class by name.

        Args:
            name (str): Unique name identifier for the policy.

        Returns:
            Optional[Type]: The policy class if found, else None.
        """
        return cls._registry.get(name.lower())

    @classmethod
    def list_policies(cls) -> list[str]:
        """List all registered policy names.

        Returns:
            list[str]: Sorted list of registered policy names.
        """
        return sorted(cls._registry.keys())

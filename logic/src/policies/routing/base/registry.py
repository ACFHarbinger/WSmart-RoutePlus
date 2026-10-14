"""
Routing Algorithm Registry Module.

This module provides a central registry for all routing algorithm implementations,
allowing for dynamic discovery and instantiation of routing algorithms by string names.

Attributes:
    RoutingAlgorithmRegistry: The singleton registry class.

Example:
    >>> from logic.src.policies.routing.base import RoutingAlgorithmRegistry
    >>> @RoutingAlgorithmRegistry.register("my_routing_algorithm")
    ... class MyPolicy(IRoutingAlgorithm): ...
    >>> policy_cls = RoutingAlgorithmRegistry.get("my_policy")
"""

from typing import Callable, Dict, List, Optional, Type

from logic.src.interfaces.routing_algorithm import IRoutingAlgorithm


# --- Routing Algorithm Registry ---
class RoutingAlgorithmRegistry:
    """
    Central registry for routing algorithms.

    Attributes:
        _registry (Dict[str, Type[IRoutingAlgorithm]]): Internal dictionary mapping policy names to classes.
    """

    _registry: Dict[str, Type[IRoutingAlgorithm]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a policy class.

        Args:
            name: Unique identifier for the policy.

        Returns:
            Callable: The decorator function.
        """

        def decorator(policy_cls: Type[IRoutingAlgorithm]) -> Type[IRoutingAlgorithm]:
            """
            Inner decorator function.

            Args:
                policy_cls: The policy implementation class.

            Returns:
                Type[IRoutingAlgorithm]: The registered policy class.
            """
            cls._registry[name] = policy_cls
            return policy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[IRoutingAlgorithm]]:
        """
        Retrieve a policy class by name.

        Args:
            name: The name of the policy to retrieve.

        Returns:
            Optional[Type[IRoutingAlgorithm]]: The policy class if found, else None.
        """
        return cls._registry.get(name)

    @classmethod
    def list_routing_algorithms(cls) -> List[str]:
        """
        List all registered routing algorithms.

        Returns:
            List[str]: A list of names of all registered routing algorithms.
        """
        return list(cls._registry.keys())

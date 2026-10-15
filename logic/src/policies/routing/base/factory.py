"""Factory for routing algorithms. Construction is ``ROUTING.create``."""

from typing import Any, Optional

from logic.src.interfaces.routing_algorithm import IRoutingAlgorithm
from logic.src.utils.registry import ROUTING

from .registry import RoutingAlgorithmRegistry


class RoutingAlgorithmFactory:
    """Create a routing adapter by registry name."""

    @classmethod
    def ensure_registered(cls) -> None:
        """Manifest entries are loaded when the registry module is imported."""
        return None

    @staticmethod
    def get_adapter(
        name: str,
        config: Optional[dict] = None,
        engine: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> IRoutingAlgorithm:
        if not isinstance(name, str):
            raise TypeError(f"Policy name must be a string, got {type(name)}")
        del engine, threshold, kwargs
        key = name.lower()
        resolved = key if RoutingAlgorithmRegistry.get(key) is not None else f"policy_{key}"
        if RoutingAlgorithmRegistry.get(resolved) is None:
            raise ValueError(f"Unknown policy: {name}. Ensure it is registered in RoutingAlgorithmRegistry.")
        return ROUTING.create(resolved, cfg=config)

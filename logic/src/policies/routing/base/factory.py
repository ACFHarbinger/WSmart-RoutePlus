"""
Policy Adapter module - Unified interface for all routing policies.

This module implements the Adapter design pattern to provide a consistent
interface for executing diverse routing policies within the simulator.

Attributes:
    RoutingAlgorithmFactory: Factory for creating routing algorithms.

Example:
    >>> from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
    >>> adapter = RoutingAlgorithmFactory.get_adapter("hgs")
"""

import importlib
import logging
import pkgutil
from typing import Any, Optional

# --- IRoutingAlgorithm Interface ---
from logic.src.interfaces.routing_algorithm import IRoutingAlgorithm

from .registry import RoutingAlgorithmRegistry

_logger = logging.getLogger(__name__)


class RoutingAlgorithmFactory:
    """
    Factory for creating routing algorithms.

    Attributes:
        _registered (bool): Internal flag indicating if policies have been registered.
    """

    _registered = False

    @classmethod
    def ensure_registered(cls) -> None:
        """Import all adapter modules to trigger @RoutingAlgorithmRegistry.register() decorators."""
        if cls._registered:
            return

        # Exact Stochastic and Decomposition Solvers
        import logic.src.policies.routing.exact_and_decomposition_solvers as exact_solvers  # noqa

        # Meta-Heuristics
        import logic.src.policies.routing.meta_heuristics as meta_heuristics  # noqa

        # Hyper-Heuristics
        import logic.src.policies.routing.hyper_heuristics as hyper_heuristics  # noqa

        # Matheuristics (Exact Solvers + Heuristics)
        import logic.src.policies.routing.matheuristics as matheuristics  # noqa

        # Learning Algorithms
        import logic.src.policies.routing.learning_algorithms as learning_algorithms  # noqa

        # Learning Heuristic Algorithms (Learning Algorithms + Heuristics)
        import logic.src.policies.routing.learning_heuristic_algorithms as learning_heuristic_algorithms  # noqa
        import logic.src.policies.routing.learning_matheuristic_algorithms as learning_matheuristic_algorithms  # noqa

        # Acceptance Criteria
        import logic.src.policies.acceptance_criteria as acceptance_criteria  # noqa

        # Other Algorithms
        import logic.src.policies.routing.other_algorithms as other_algorithms  # noqa

        # Joint Packing and Routing Algorithms
        import logic.src.policies.packing_and_routing as packing_and_routing  # noqa

        # Some family packages do not import every adapter in their __init__ (SWC-TCF, EGH, LASM,
        # CP-SAT, the stochastic/decomposition solvers, ...). Import every ``policy_*`` module so the
        # registry never depends on which modules happen to have been imported already. A module
        # whose optional solver library is missing is skipped with a warning.
        import logic.src.policies.routing as routing

        for info in pkgutil.walk_packages(routing.__path__, routing.__name__ + "."):
            if info.name.rsplit(".", 1)[-1].startswith("policy_"):
                try:
                    importlib.import_module(info.name)
                except ImportError as exc:
                    _logger.warning("Routing algorithm %s not registered: %s", info.name, exc)

        cls._registered = True

    @staticmethod
    def get_adapter(
        name: str,
        config: Optional[dict] = None,
        engine: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> IRoutingAlgorithm:
        """
        Create and return the appropriate PolicyAdapter for the given parameters.

        Args:
            name: Policy name (e.g., 'alns', 'hgs', 'tsp').
            config: Raw policy config dict from YAML. If provided, the adapter's
                    typed config dataclass is built automatically.
            engine: Deprecated. Engine should be specified in config.
            threshold: Deprecated. Threshold should be specified in config.
            kwargs: Additional keyword arguments (unused, for backward compat).


        Returns:
            Instantiated policy adapter with typed config.
        """
        RoutingAlgorithmFactory.ensure_registered()

        # Normalize name
        if not isinstance(name, str):
            raise TypeError(f"Policy name must be a string, got {type(name)}")
        name = name.lower()

        # Try Registry first
        cls = RoutingAlgorithmRegistry.get(name) or RoutingAlgorithmRegistry.get(f"policy_{name}")

        if cls:
            if config is not None:
                return cls(config=config)  # type: ignore[return-value,call-arg]
            return cls()  # type: ignore[return-value]

        raise ValueError(f"Unknown policy: {name}. Ensure it is registered in RoutingAlgorithmRegistry.")

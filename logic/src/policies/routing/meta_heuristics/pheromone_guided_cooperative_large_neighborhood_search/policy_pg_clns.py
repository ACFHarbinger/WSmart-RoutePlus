"""
PG-CLNS Policy Adapter.

Adapts the Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS) logic to the agnostic interface.

Attributes:
    PGCLNSPolicy: Policy adapter for the PG-CLNS metaheuristic.

Example:
    >>> policy = PGCLNSPolicy()
    >>> routes, profit, cost = policy(obs)
"""

from typing import Any, Dict, Optional, Type, Union

from logic.src.configs.policies.routing.meta_heuristics.pg_clns import PGCLNSConfig
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy

from .pg_clns import PGCLNSSolver


@RoutingAlgorithmRegistry.register("pg_clns")
class PGCLNSPolicy(GenericRoutingPolicy):
    """
    PG-CLNS policy class.

    Visits pre-selected 'mandatory' bins using the population-based PG-CLNS metaheuristic.

    PG-CLNS is a hybrid metaheuristic that combines the constructive power of
    Ant Colony Optimization (ACO) with the local improvement capabilities of
    Large Neighborhood Search (LNS) within a cooperative population-based framework.

    Attributes:
        config: Configuration parameters for the policy.
    """

    _solver = PGCLNSSolver

    def __init__(self, config: Optional[Union[PGCLNSConfig, Dict[str, Any]]] = None):
        """Initialize PG-CLNS policy with optional config.

        Args:
            config: PGCLNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for PG-CLNS.

        Returns:
            Optional[Type]: The PGCLNSConfig class.
        """
        return PGCLNSConfig

    def _get_config_key(self) -> str:
        """Return config key for PG-CLNS.

        Returns:
            str: "pg_clns".
        """
        return "pg_clns"

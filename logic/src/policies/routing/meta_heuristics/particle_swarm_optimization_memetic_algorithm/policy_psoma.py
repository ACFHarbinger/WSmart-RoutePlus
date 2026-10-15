"""
PSOMA Policy Adapter.

Adapts the Particle Swarm Optimization Memetic Algorithm (PSOMA) to the
agnostic BaseRoutingPolicy interface.

Attributes:
    PSOMAConfig (Type): Configuration schema for the PSOMA solver.
    PSOMAPolicy (Type): Policy adapter for the PSOMA solver.
    GenericRoutingPolicy (Type): Declarative adapter base (WP-11 P-30).

Example:
    >>> from logic.src.configs.policies.psoma import PSOMAConfig
    >>> config = PSOMAConfig(pop_size=20)
    >>> policy = PSOMAPolicy(config)
    >>> routes = policy.solve(problem)
"""

from typing import Any, Dict, Optional, Type, Union

from logic.src.configs.policies.psoma import PSOMAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy

from .solver import PSOMASolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.SWARM_INTELLIGENCE,
    PolicyTag.MEMETIC_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("psoma", config=PSOMAConfig)
class PSOMAPolicy(GenericRoutingPolicy):
    """
    PSOMA policy class.

    Visits bins using Particle Swarm Optimization with a memetic local-search step.
    The generic adapter materializes the effective PSOMAConfig, builds the
    SubProblem and runs the declared ``PSOMASolver`` engine.

    Attributes:
        _solver (PSOMASolver): Engine instantiated and solved by the generic path.
    """

    _solver = PSOMASolver

    def __init__(self, config: Optional[Union[PSOMAConfig, Dict[str, Any]]] = None):
        """Initializes the PSOMA policy.

        Args:
            config (Optional[Union[PSOMAConfig, Dict[str, Any]]]): Configuration
                source for the Particle Swarm Optimization Memetic Algorithm.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for PSOMA.

        Returns:
            Optional[Type]: The PSOMAConfig class.
        """
        return PSOMAConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the PSOMA policy.

        Returns:
            str: The registry key 'psoma'.
        """
        return "psoma"

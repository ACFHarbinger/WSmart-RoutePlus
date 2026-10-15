r"""ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.

Attributes:
    ALNSPolicy: Policy class for ALNS.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search import ALNSPolicy
    >>> policy = ALNSPolicy()
"""

from typing import Any, Dict, Optional, Type, Union

from logic.src.configs.policies import ALNSConfig
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy
from logic.src.policies.routing.base.sub_problem import SubProblem

from .dispatcher import run_alns


@RoutingAlgorithmRegistry.register("alns")
class ALNSPolicy(GenericRoutingPolicy):
    r"""ALNS policy class.

    Visits pre-selected 'mandatory' bins using Adaptive Large Neighborhood Search.

    ALNS is a powerful metaheuristic that iteratively improves a solution by
    applying a sequence of destruction (removal) and reconstruction
    (insertion) operators. The selection of operators is controlled by a
    roulette-wheel mechanism that adapts based on the historical performance
    of each operator in discovering high-quality solutions.

    Attributes:
        config: Configuration for the policy.

    Example:
        >>> policy = ALNSPolicy()
    """

    _solver = run_alns

    def __init__(self, config: Optional[Union[ALNSConfig, Dict[str, Any]]] = None):
        """Initialize ALNS policy with optional config.

        Args:
            config: ALNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for ALNS.

        Returns:
            The ALNSConfig class.
        """
        return ALNSConfig

    def _get_config_key(self) -> str:
        """Return config key for ALNS.

        Returns:
            The string "alns".
        """
        return "alns"

    def _solve(self, problem: SubProblem, cfg: Any) -> Any:
        """Run the ALNS dispatcher with the policy's telemetry recorder.

        Args:
            problem: The routing sub-problem.
            cfg: The effective ALNSConfig.

        Returns:
            SolveResult: Best routes, profit, and cost.
        """
        problem.recorder = self._viz
        return type(self)._solver(problem, cfg)

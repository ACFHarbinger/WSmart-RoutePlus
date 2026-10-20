"""MS-BPC-SP Policy Adapter.

Adapts the Multi-Stage Branch-and-Price-and-Cut with Set Partitioning
(MS-BPC-SP) engine to the agnostic interface.

Attributes:
    MSBPCSPPolicy (class): Policy wrapper for the MS-BPC-SP solver.

Example:
    >>> policy = MSBPCSPPolicy(config=cfg)
    >>> routes, profit, cost = policy.execute(env, bins)
"""

from typing import Any, Dict, Optional, Type, Union

from logic.src.configs.policies import MSBPCSPConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy
from logic.src.utils.registry import ALL

from .ms_bpc_sp_engine import run_ms_bpc_sp


@ALL.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ms_bpc_sp")
class MSBPCSPPolicy(GenericRoutingPolicy):
    """Multi-stage Branch-and-Price-and-Cut with Set Partitioning policy class.

    Visits pre-selected 'mandatory' bins using exact or heuristic BPC solvers.

    Attributes:
        config (MSBPCSPConfig): Configuration object.
    """

    _solver = run_ms_bpc_sp

    def __init__(self, config: Optional[Union[MSBPCSPConfig, Dict[str, Any]]] = None):
        """Initialize MSBPCSP policy with optional config.

        Args:
            config: MSBPCSPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for MS-BPC-SP.

        Returns:
            Type: MSBPCSPConfig class.
        """
        return MSBPCSPConfig

    def _get_config_key(self) -> str:
        """Return config key for MS-BPC-SP.

        Returns:
            str: "ms_bpc_sp".
        """
        return "ms_bpc_sp"

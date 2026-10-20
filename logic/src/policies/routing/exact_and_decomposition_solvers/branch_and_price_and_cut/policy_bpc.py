"""BPC Policy Adapter.

Adapts the Branch-and-Price-and-Cut (BPC) logic to the agnostic interface.

Attributes:
    BPCPolicy (class): Policy wrapper for the BPC solver.

Example:
    >>> policy = BPCPolicy(config=bpc_cfg)
    >>> routes, profit, cost = policy.execute(env, bins)
"""

from typing import Any, Dict, Optional, Type, Union

from logic.src.configs.policies import BPCConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
    run_bpc,
)
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("bpc")
class BPCPolicy(GenericRoutingPolicy):
    """Branch-and-Price-and-Cut policy class.

    Visits pre-selected 'mandatory' bins using exact or heuristic BPC solvers.

    Attributes:
        config (BPCConfig): Configuration object.
    """

    _solver = run_bpc

    def __init__(self, config: Optional[Union[BPCConfig, Dict[str, Any]]] = None):
        """Initialize BPC policy with optional config.

        Args:
            config: BPCConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for BPC.

        Returns:
            Type: BPCConfig class.
        """
        return BPCConfig

    def _get_config_key(self) -> str:
        """Return config key for BPC.

        Returns:
            str: "bpc".
        """
        return "bpc"

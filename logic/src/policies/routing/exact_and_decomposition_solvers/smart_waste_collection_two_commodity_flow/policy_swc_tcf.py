r"""SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Policy Wrapper.

Attributes:
    SWCTCFPolicy: Simulator policy adapter for TCF.

Example:
    >>> policy = SWCTCFPolicy()
    >>> routes, profit, cost = policy._run_solver(dist, wastes, cap, rev, cost, values, mandatory)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from logic.src.configs.policies import SWCTCFConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy
from logic.src.policies.routing.base.sub_problem import SubProblem
from logic.src.utils.registry import ALL

from .dispatcher import run_swc_tcf_optimizer


@ALL.register(
    PolicyTag.EXACT,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("swc_tcf")
class SWCTCFPolicy(GenericRoutingPolicy):
    r"""Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy adapter.

    This policy implements a mathematical programming approach based on the
    Two-Commodity Flow formulation for the Vehicle Routing Problem. It allows
    for the use of Gurobi (exact) as the underlying optimization engine.

    Technical Context:
    - Formulation: Uses a flow-based MILP model to ensure subtour elimination
      and capacity enforcement.
    - Dispatcher: Multi-backend dispatcher supporting exact and heuristic engines.

    Reference:
    - Ramos, T. R. P., Morais, C. S., & Barbosa-Povoa, A. P. (2018). "The smart
      waste collection routing problem: Alternative operational management
      approaches". Expert Systems with Applications.

    Attributes:
        config (SWCTCFConfig): Policy configuration.
    """

    _solver = run_swc_tcf_optimizer

    def __init__(self, config: Optional[Union[SWCTCFConfig, Dict[str, Any]]] = None):
        """Initialize SWC-TCF policy with optional config.

        Args:
            config: SWCTCFConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for this policy.

        Returns:
            Optional[Type[SWCTCFConfig]]: The configuration class.
        """
        return SWCTCFConfig

    def _get_config_key(self) -> str:
        """Return the configuration key.

        Returns:
            str: The config key.
        """
        return "swc_tcf"

    def _finalize(self, result: Any, problem: SubProblem) -> Tuple[Any, float, float]:
        """Split the flat depot-delimited route into per-route node lists.

        The backends return a flat depot-delimited list
        ([0, route-1 bins..., 0, route-2 bins..., 0]); split on the depot
        marker so multi-route days keep their route boundaries.
        """
        raw_route, profit, cost = result
        routes: List[List[int]] = []
        current: List[int] = []
        for node in raw_route:
            if node == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            routes.append(current)
        return routes, profit, cost

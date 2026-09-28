r"""SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Policy Wrapper.

Attributes:
    SWCTCFPolicy: Simulator policy adapter for TCF.

Example:
    >>> policy = SWCTCFPolicy()
    >>> res = policy._run_solver(dist, wastes, cap, rev, cost, values, mandatory)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SWCTCFConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .dispatcher import run_swc_tcf_optimizer
from .params import SWCTCFParams


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("swc_tcf")
class SWCTCFPolicy(BaseRoutingPolicy):
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

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute core Smart Waste Collection - Two-Commodity Flow (SWC-TCF) solver.

        This implementation adapts the flow-based MILP model to the standardized
        agnostic routing interface. It resolves backends (Gurobi, Pyomo, OR-Tools)
        and handles the translation of local sub-problem data into solver-specific
        formats.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the sub-problem.
            sub_wastes: Mapping of node indices to fill percentages (0..100).
            capacity: Vehicle capacity in percentage points (relative to 100% per bin).
            revenue: Euro per percentage point of fill collected.
            cost_unit: Euro per kilometer traveled.
            values: Merged configuration dictionary from Policy and defaults.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[List[List[int]], float, float]: (routes, profit, travel_cost)
        """
        # 1. Prepare inputs for the dispatcher
        # sub_wastes is {idx: fill}. Dispatcher expects a list/array.
        n_nodes = len(sub_wastes)
        amounts = np.zeros(n_nodes)
        for i, fill in sub_wastes.items():
            amounts[i - 1] = fill  # Solver uses 1-based binsids internally for depot mapping

        binsids = list(range(1, n_nodes + 1))

        # 2. Extract and sanitize parameters
        params = SWCTCFParams.from_config(values)
        seed = kwargs.get("seed") if kwargs.get("seed") is not None else getattr(self, "_seed", params.seed)

        # Update values with the standardized ones from the adapter
        values_to_pass = {**values, "Q": capacity, "R": revenue, "C": cost_unit}

        # 3. Invoke dispatcher
        # run_swc_tcf_optimizer returns Tuple[List[int], float, float] -> (route, profit, cost)
        # Note: route is customer sequence (0-depot-0 managed by factory/base).
        raw_route, profit, cost = run_swc_tcf_optimizer(
            bins=amounts,
            distance_matrix=sub_dist_matrix.tolist(),
            values=values_to_pass,
            binsids=binsids,
            mandatory_nodes=mandatory_nodes,
            # Fleet limit from the simulation context (sim.n_vehicles); 0 or
            # absent means unbounded, matching the other constructors. A plain
            # default of 1 would silently cap every day at one vehicle.
            number_vehicles=(
                int(nv) if (nv := kwargs.get("n_vehicles", kwargs.get("number_vehicles"))) and int(nv) > 0 else 0
            ),
            time_limit=float(params.time_limit),  # int() turned sub-second budgets into "no limit"
            framework=params.framework,
            optimizer=params.engine,
            seed=int(seed) if seed is not None else 42,
            dual_values=kwargs.get("dual_values"),
        )

        # 4. Normalize route to List[List[int]] format expected by _run_solver signature.
        # The backends return a flat depot-delimited list
        # ([0, route-1 bins..., 0, route-2 bins..., 0]); split on the depot
        # marker so multi-route days keep their route boundaries.
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

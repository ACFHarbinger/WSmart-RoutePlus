r"""LBBD → ALNS → BPC → RL → SP Pipeline Policy.

Adapts the five-stage pipeline to the ``BaseRoutingPolicy`` interface and
registers it in the routing-algorithm registry as ``"lasm"``.

Quality / speed dial
--------------------
A single ``alpha ∈ [0, 1]`` parameter in the YAML controls default budgets.
The RL controller will override these as it accumulates experience.

Example YAML configuration::

    lasm:
      custom:
        - alpha: 0.5
        - time_limit: 120.0
        - rl_mode: online
        - rl_policy_path: /tmp/lbbd_rl_policy.json
        - lbbd_sub_solver: alns
        - lbbd_cut_families: [nogood, optimality, pareto]
        - lbbd_max_iterations: 20
        - skip_bpc: false
        - alns_profit_aware_operators: true
        - bpc_cutting_planes: rcc
        - sp_pool_cap: 50000

Attributes:
    LASMPipelinePolicy: Policy class.

Example:
    >>> policy = LASMPipelinePolicy()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.lasm import LASMPipelineConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.registry import ALL

from .dispatcher import run_lasm_pipeline


@ALL.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.EXACT,
    PolicyTag.META_HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("lasm")
class LASMPolicy(BaseRoutingPolicy):
    r"""Five-stage VRPP pipeline: LBBD → ALNS → BPC → RL → SP.

    Attributes:
        config: Configuration object or dict.
    """

    def __init__(
        self,
        config: Optional[Union["LASMPipelineConfig", Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the pipeline policy.

        Args:
            config: LASMPipelineConfig, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for this policy.

        Returns:
            Optional[Type]: The configuration class.
        """
        return LASMPipelineConfig

    def _get_config_key(self) -> str:
        """Return the configuration key for this policy.

        Returns:
            str: The configuration key.
        """
        return "lasm"

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
        """Execute the five-stage LBBD pipeline.

        Args:
            sub_dist_matrix: (n+1) × (n+1) local distance matrix.
            sub_wastes:      {local_node_id → fill_level}.
            capacity:        Vehicle capacity Q.
            revenue:         Revenue per unit waste R.
            cost_unit:       Cost per unit distance C.
            values:          Merged YAML config dict.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs:          n_vehicles, model_env, recorder, …

        Returns:
            (routes, profit, cost)
        """
        n_bins = sub_dist_matrix.shape[0] - 1
        bins = np.array([sub_wastes.get(i, 0.0) for i in range(1, n_bins + 1)])
        binsids = list(range(1, n_bins + 1))

        problem_values: Dict[str, Any] = {
            "Q": capacity,
            "R": revenue,
            "C": cost_unit,
            "Omega": values.get("Omega", 0.0),
            "delta": values.get("delta", 0.0),
            "psi": values.get("psi", 0.99),
            "B": values.get("B", 0),
            "V": values.get("V", 1),
        }

        p = self._effective_config(values)
        if "time_limit" in values and values["time_limit"] > 0:
            p = LASMPipelineConfig.from_config({**values})

        n_vehicles: int = kwargs.get("n_vehicles") or 1
        env = kwargs.get("model_env")
        recorder = getattr(self, "_viz", None)

        flat_route, profit, cost = run_lasm_pipeline(
            bins=bins,
            dist_matrix=sub_dist_matrix.tolist(),
            env=env,
            values=problem_values,
            binsids=binsids,
            mandatory=mandatory_nodes,
            n_vehicles=n_vehicles,
            params=p,
            recorder=recorder,
        )

        # Convert flat route to list-of-lists
        routes: List[List[int]] = []
        current: List[int] = []
        for node in flat_route[1:]:
            if node == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            routes.append(current)

        return routes, profit, cost

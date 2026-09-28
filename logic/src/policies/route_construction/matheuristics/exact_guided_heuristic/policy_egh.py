r"""TCF → ALNS → BPC → SP-merge Pipeline Policy.

Adapts the four-stage pipeline to the ``BaseRoutingPolicy`` interface so it
can be registered in the route-constructor registry and used interchangeably
with all other policies (ALNS, BPC, SWC-TCF, PSOMA, …).

Quality / speed dial
--------------------
The pipeline exposes a single ``alpha`` parameter in the YAML:

    alpha: 0.0  →  TCF + tiny ALNS only  (fastest, no BPC)
    alpha: 0.5  →  balanced              (default)
    alpha: 1.0  →  full BPC + large ALNS (highest quality, slowest)

Example YAML configuration::

    pipeline:
      custom:
        - alpha: 0.5
        - time_limit: 120.0
        - skip_bpc: false
        - alns_profit_aware_operators: true
        - alns_extended_operators: false
        - bpc_cutting_planes: rcc
        - sp_pool_cap: 50000

Attributes:
    PipelinePolicy: Policy class for the four-stage VRPP pipeline.

Example:
    >>> policy = PipelinePolicy()
    >>> routes, profit, cost = policy.execute(env, bins)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ExactGuidedHeuristicConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .dispatcher import run_pipeline
from .params import ExactGuidedHeuristicParams


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.EXACT,
    PolicyTag.META_HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("egh")
class ExactGuidedHeuristicPolicy(BaseRoutingPolicy):
    r"""Exact Guided Heuristic (EGH) VRPP policy: TCF → ALNS → BPC → SP-merge.

    Integrates exact (BPC, SP), flow-formulation (TCF), and metaheuristic
    (ALNS) components into a single tunable solver.  The quality/speed
    trade-off is controlled by ``alpha ∈ [0, 1]``.

    Attributes:
        config: Configuration object or dict.

    Example:
        >>> policy = PipelinePolicy()
    """

    def __init__(
        self,
        config: Optional[Union["ExactGuidedHeuristicConfig", Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the pipeline policy.

        Args:
            config: PipelineConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class.

        Returns:
            ExactGuidedHeuristicConfig (or None if not yet defined in the config package).
        """
        return ExactGuidedHeuristicConfig

    def _get_config_key(self) -> str:
        """Return the YAML config key for this policy.

        Returns:
            The string "egh".
        """
        return "egh"

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
        """Execute the four-stage pipeline.

        The ``values`` dict is the merged YAML configuration already flattened
        by ``BaseRoutingPolicy``.  ``ExactGuidedHeuristicParams.from_config`` tolerates
        extra keys, so no pre-filtering is needed.

        Args:
            sub_dist_matrix: (n+1) × (n+1) local distance matrix.
            sub_wastes:      {local_node_id → fill_level}.
            capacity:        Vehicle capacity Q.
            revenue:         Revenue per unit waste R.
            cost_unit:       Cost per unit distance C.
            values:          Merged YAML config dict.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs:          Extra context (n_vehicles, model_env, recorder, …).

        Returns:
            (routes, profit, cost)
                routes — list of customer-node lists (local indices, no depot).
                profit — net profit.
                cost   — total travel cost.
        """
        # ── Reconstruct inputs expected by the pipeline dispatcher ─────────
        # sub_wastes has local indices 1..n; build the bins array and binsids
        # list that the TCF stage expects.
        n_bins = sub_dist_matrix.shape[0] - 1
        bins = np.array([sub_wastes.get(i, 0.0) for i in range(1, n_bins + 1)])

        # binsids: use the local 1..n indices directly (global mapping is
        # handled at a higher level by BaseRoutingPolicy).
        binsids = list(range(1, n_bins + 1))

        # Rebuild problem-parameter dict in the format expected by the TCF stage
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

        # Pipeline-specific parameters from the merged config
        p = ExactGuidedHeuristicParams.from_config(values)
        # Inject time_limit from values if present (BaseRoutingPolicy sets it)
        if "time_limit" in values and values["time_limit"] > 0:
            p = ExactGuidedHeuristicParams.from_config({**values.get("pipeline", {}), **values})

        n_vehicles: int = kwargs.get("n_vehicles") or 1
        env = kwargs.get("model_env")
        recorder = getattr(self, "_viz", None)

        flat_route, profit, cost = run_pipeline(
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

        # ── Convert flat route back to list-of-lists ───────────────────────
        # flat_route is [0, c1, c2, 0, c3, c4, c5, 0, …] with 0 as separator.
        routes: List[List[int]] = []
        current_route: List[int] = []
        for node in flat_route[1:]:  # skip leading depot
            if node == 0:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        if current_route:
            routes.append(current_route)

        return routes, profit, cost

"""BPC Policy Adapter.

Adapts the Branch-and-Price-and-Cut (BPC) logic to the agnostic interface.

Attributes:
    BPCPolicy (class): Policy wrapper for the BPC solver.

Example:
    >>> policy = BPCPolicy(config=bpc_cfg)
    >>> routes, profit, cost = policy.execute(env, bins)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MSBPCSPConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .ms_bpc_sp_engine import run_ms_bpc_sp
from logic.src.configs.policies.ms_bpc_sp import MSBPCSPConfig


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ms_bpc_sp")
class MSBPCSPPolicy(BaseRoutingPolicy):
    """Multi-stage Branch-and-Price-and-Cut with Set Partitioning policy class.

    Visits pre-selected 'mandatory' bins using exact or heuristic BPC solvers.

    Attributes:
        config (MSBPCSPConfig): Configuration object.
    """

    def __init__(self, config: Optional[Union[MSBPCSPConfig, Dict[str, Any]]] = None):
        """Initialize MSBPCSP policy with optional config.

        Args:
            config: MSBPCSPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for BPC.

        Returns:
            Type: MSBPCSPConfig class.
        """
        return MSBPCSPConfig

    def _get_config_key(self) -> str:
        """Return config key for BPC.

        Returns:
            str: "ms_bpc_sp".
        """
        return "ms_bpc_sp"

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
        """Execute core MS-BPC-SP optimization.

        Args:
            sub_dist_matrix: Symmetric distance matrix.
            sub_wastes: Map of local node indices to waste volume.
            capacity: Vehicle capacity.
            revenue: Revenue per kg.
            cost_unit: Cost per km.
            values: Configuration dictionary.
            mandatory_nodes: Local indices of nodes that must be visited.
            kwargs: Additional parameters (n_vehicles, model_env, etc).

        Returns:
            A 3-tuple of (routes, profit, cost).
        """
        # Return contract for run_bpc:
        #   routes          — list of customer-node lists (depot excluded)
        #   objective_value — net profit = Σ(revenue_i) - travel_cost, in monetary units.
        #                     May be a greedy-fallback value if BPC found no integer solution.
        # Convert local mandatory indices to a set of mandatory nodes for the solver
        mandatory_indices: Set[int] = set(mandatory_nodes)

        # Initialize standardized params object (Phase 1 refactoring)
        params = self._effective_config(values)

        # Extract vehicle limit from simulation context (sim.n_vehicles)
        n_vehicles = kwargs.get("n_vehicles")
        # Explicit int conversion and positive check. None and 0 both map to
        # unlimited fleet. False is rejected at the int() call (TypeError surfaced
        # to the caller) rather than silently treated as unlimited.
        vehicle_limit = None if n_vehicles is None else int(n_vehicles) if int(n_vehicles) > 0 else None

        # run_bpc returns (routes, objective_value) where objective_value is
        # net profit (revenue - travel_cost) in the problem's monetary units.
        # It is NOT a raw travel cost despite the variable name used in run_bpc's
        # return signature. Rename immediately to prevent future misreading.
        routes, objective_value = run_ms_bpc_sp(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_indices=mandatory_indices,
            vehicle_limit=vehicle_limit,
            env=kwargs.get("model_env"),
            node_coords=kwargs.get("node_coords"),
            recorder=kwargs.get("recorder"),
        )

        profit = objective_value

        # Compute raw travel distance (km)
        raw_distance = 0.0
        for route in routes:
            # Normalize: strip any leading/trailing depot index before wrapping.
            # Route.nodes stores customer-only sequences, but defensive stripping
            # guards against representation changes in run_bpc's return value.
            inner = [n for n in route if n != 0]
            if not inner:
                continue

            path = [0] + inner + [0]
            for i in range(len(path) - 1):
                raw_distance += sub_dist_matrix[path[i]][path[i + 1]]

        return routes, profit, raw_distance

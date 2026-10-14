"""Joint Greedy Orienteering Policy.

A constructive heuristic policy for joint packing and routing, picking nodes
based on their incremental profitability using a randomized greedy approach.

Attributes:
    JointGreedyPolicy: The policy class implementing JGO logic.

Example:
    >>> policy = JointGreedyPolicy()
    >>> selection, routes, profit, cost = policy.solve_joint(context)
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.jgo import JointGreedyConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.joint_context import JointSelectionConstructionContext
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.packing_and_routing.base.base_joint_policy import BaseJointPolicy
from logic.src.policies.packing_and_routing.base.registry import JointPolicyRegistry

from .params import JointGreedyParams


@GlobalRegistry.register(
    PolicyTag.HEURISTIC,
    PolicyTag.PACKING,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.SINGLE_PERIOD,
    PolicyTag.PACKING_AND_ROUTING,
)
@JointPolicyRegistry.register("jgo")
@RoutingAlgorithmRegistry.register("jgo")
class JointGreedyPolicy(BaseJointPolicy):
    """Greedy constructive policy for joint packing and routing.

    Uses a multi-start randomized greedy heuristic to balance local profitability
    (revenue minus distance cost) against resource consumption.

    Attributes:
        config (Optional[Union[JointGreedyConfig, Dict[str, Any]]]): Configuration source.
    """

    def __init__(self, config: Optional[Union[JointGreedyConfig, Dict[str, Any]]] = None):
        """
        Initialize the JGO policy.

        Args:
            config (Optional[Union[JointGreedyConfig, Dict[str, Any]]]): Configuration source.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the dataclass type for config parsing.

        Returns:
            Optional[Type]: The JointGreedyConfig type.
        """
        return JointGreedyConfig

    def _get_config_key(self) -> str:
        """Return the Hydra configuration key.

        Returns:
            str: The configuration key string 'jgo'.
        """
        return "jgo"

    def solve_joint(
        self,
        context: JointSelectionConstructionContext,
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """Run multi-start randomized greedy construction.

        Args:
            context (JointSelectionConstructionContext): The search and problem context.

        Returns:
            Tuple[List[int], List[List[int]], float, float]:
                - selected_bins: List of unique bin IDs selected.
                - local_routes: Constructed routes (local node indices).
                - total_profit: Net profit (Revenue - Distance Cost).
                - total_cost: Total routing cost.
        """
        params = JointGreedyParams.from_config(self._config) if self._config else JointGreedyParams()

        start_time = time.time()
        random.seed(params.seed)
        np.random.seed(params.seed)

        best_selection: List[int] = []
        best_routes: List[List[int]] = []
        best_profit = -float("inf")
        best_cost = 0.0

        for _ in range(params.n_starts):
            if (time.time() - start_time) > params.time_limit:
                break

            selection, routes, profit, cost = self._single_greedy_run(context, params)
            if profit > best_profit:
                best_selection = selection
                best_routes = routes
                best_profit = profit
                best_cost = cost

        return best_selection, best_routes, best_profit, best_cost

    def _single_greedy_run(
        self, context: JointSelectionConstructionContext, params: JointGreedyParams
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """One randomized greedy construction pass.

        Args:
            context (JointSelectionConstructionContext): The problem data.
            params (JointGreedyParams): Solver configurations.

        Returns:
            Tuple[List[int], List[List[int]], float, float]:
                Partial or full selection and routing solution.
        """
        n_bins = len(context.bin_ids)
        unvisited = list(range(n_bins))

        selected_bins: List[int] = []
        routes: List[List[int]] = []
        total_cost = 0.0
        revenue_scaled = context.revenue_scaled()

        while unvisited:
            route_global_ids: List[int] = []
            curr_cap = context.capacity
            curr_loc = 0

            while unvisited:
                candidates = []
                for i in unvisited:
                    mass = context.current_fill[i] * (context.bin_density * context.bin_volume / 100.0)
                    if mass <= curr_cap:
                        dist_to = context.distance_matrix[curr_loc, context.bin_ids[i]]
                        revenue_node = context.current_fill[i] * revenue_scaled
                        cost_node = dist_to * context.cost_per_km

                        score = (revenue_node - cost_node) / max(dist_to, 0.001)
                        candidates.append((i, score, dist_to, mass))

                if not candidates:
                    break

                candidates.sort(key=lambda x: x[1], reverse=True)
                top_k = candidates[: params.k_best]
                chosen_idx_in_candidates = random.randrange(len(top_k))
                chosen_idx, _, chosen_dist, chosen_mass = top_k[chosen_idx_in_candidates]

                route_global_ids.append(context.bin_ids[chosen_idx])
                selected_bins.append(context.bin_ids[chosen_idx])
                curr_cap -= chosen_mass
                curr_loc = context.bin_ids[chosen_idx]
                total_cost += chosen_dist
                unvisited.remove(chosen_idx)

            if route_global_ids:
                total_cost += context.distance_matrix[curr_loc, 0]
                routes.append(route_global_ids)
            else:
                break

        total_mass = 0.0
        for bid in selected_bins:
            idx = np.where(context.bin_ids == bid)[0][0]
            total_mass += context.current_fill[idx] * (context.bin_density * context.bin_volume / 100.0)

        total_profit = (total_mass * context.revenue_kg) - (total_cost * context.cost_per_km)

        subset_indices = [0] + selected_bins
        global_to_local = {gbid: lidx for lidx, gbid in enumerate(subset_indices)}

        local_routes = []
        for rgids in routes:
            local_routes.append([global_to_local[gid] for gid in rgids])

        return selected_bins, local_routes, total_profit, total_cost

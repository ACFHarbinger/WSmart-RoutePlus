"""Execution template of the base routing policy.

:class:`RoutingExecutionMixin` holds the
:class:`~logic.src.policies.routing.base.base_routing_policy.BaseRoutingPolicy` methods that
run one routing call: ``execute`` (mandatory validation, parameter loading, subset
problem, solver call, tour mapping and cost) and its helpers for building the subset
problem, mapping the tour back to global bin ids, computing the cost and logging the
solver parameters. Moved unchanged from ``base_routing_policy.py``.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.interfaces.context.search_context import ConstructionMetrics, SearchContext, merge_context
from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.tracking.core.run import get_active_run


class RoutingExecutionMixin:
    """The ``execute`` template and its subset-problem, tour-mapping, cost and logging helpers (moved unchanged from ``base_routing_policy.py``)."""

    def _create_subset_problem(
        self,
        mandatory: List[int],
        distance_matrix: Any,
        bins: Any,
        use_all_bins: bool = True,
    ) -> Tuple[np.ndarray, Dict[int, float], List[int], List[int]]:
        """
        Create a subset distance matrix and wastes dict for the solver.

        Args:
            mandatory: List of 1-based global bin IDs to visit.
            distance_matrix: Full distance matrix (list or ndarray).
            bins: Bins object with `.c` attribute (0-indexed fill levels).
            use_all_bins: If True, include all bins in the problem (VRPP mode).

        Returns:
            Tuple containing:
                - sub_dist_matrix (np.ndarray): Distance matrix for subset.
                - sub_wastes (Dict[int, float]): {local_idx: fill} for local nodes.
                - subset_indices (List[int]): Local to global index mapping.
                - mandatory_nodes (List[int]): Local indices that MUST be visited.
        """

        if use_all_bins:
            # Include ALL bins in the solver problem (VRPP mode)
            n_bins = len(bins.c)
            subset_indices = [0] + list(range(1, n_bins + 1))
            # In VRPP mode, mandatory nodes are those in the mandatory list.
            # Since subset_indices is [0, 1, 2, ..., N], global ID g maps to local index g.
            mandatory_nodes = list(mandatory)
        else:
            # Restrict solver ONLY to mandatory bins
            # subset_indices[0] = depot (0), subset_indices[1..M] = mandatory bins
            subset_indices = [0] + list(mandatory)
            # All nodes in this restricted subset are mandatory
            mandatory_nodes = list(range(1, len(subset_indices)))

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Build local wastes: local index i (1-based) -> fill level
        sub_wastes = {}
        for i in range(1, len(subset_indices)):
            global_idx = subset_indices[i]
            # global_idx is 1-based, bins.c is 0-indexed
            fill = bins.c[global_idx - 1]
            sub_wastes[i] = float(fill)

        return sub_dist_matrix, sub_wastes, subset_indices, mandatory_nodes

    def _map_tour_to_global(self, routes: List[List[int]], subset_indices: List[int]) -> List[int]:
        """
        Map solver routes (local indices) back to global bin IDs.

        Args:
            routes: List of routes from solver, each a list of local node indices.
            subset_indices: Mapping from local index to global index.

        Returns:
            Flat tour list with depot (0) separating routes.
        """

        tour = [0]
        if routes:
            for route in routes:
                for node_idx in route:
                    tour.append(subset_indices[node_idx])
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        return tour

    def _compute_cost(self, distance_matrix: Any, tour: List[int]) -> float:
        """
        Compute total tour cost.

        Args:
            distance_matrix: Full distance matrix.
            tour: Tour as list of node indices.

        Returns:
            Total distance cost of the tour.
        """

        from logic.src.utils.routing.tours import (
            get_route_cost,
        )

        return get_route_cost(distance_matrix, tour)

    def _log_solver_params(self, values: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Log the resolved solver parameters to the active tracking run.

        Captures the full merged ``values`` dict (area defaults + YAML config +
        runtime overrides) as run params, namespaced under the policy name.
        Only logs once per adapter instance to avoid per-day duplication.

        Args:
            values: Merged solver configuration dict from ``_load_area_params``.
            context: Simulation day context kwargs.

        """

        if getattr(self, "_params_logged", False):
            return
        self._params_logged = True

        run = get_active_run()
        if run is None:
            return

        policy_name = context.get("policy_name", self._get_config_key())
        sample_id = context.get("sample_id", 0)
        prefix = f"policy_params/{policy_name}/s{sample_id}"

        params: Dict[str, Any] = {}
        for k, v in values.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                params[f"{prefix}/{k}"] = v

        if params:
            run.log_params(params)

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[Any]]:
        """Execute the routing policy using template method pattern.

        Threads the ``SearchContext`` (from Phase 1) forward:
        1. Reads ``kwargs["search_context"]`` if present.
        2. Merges ``ConstructionMetrics`` into it after the solver runs.
        3. Returns (tour, cost, profit, search_context, multi_day_context).

        Args:
            kwargs: Simulation context containing:

                - mandatory (List[int]): Global 1-based IDs to visit.
                - bins: Bins state object.
                - distance_matrix: Full NxN matrix.
                - area (str): Geographic area name.
                - waste_type (str): Waste type (e.g., "plastic").
                - config (dict): Policy configuration overrides.
                - search_context (SearchContext): Incoming metrics ledger.
                - multi_day_context (MultiDayContext): Incoming multi-day state.
                - coords (DataFrame): Optional node coordinates for pruning.

        Returns:
            A 5-tuple of (tour, cost, profit, search_context, multi_day_context).
        """

        from logic.src.interfaces.context.search_context import SearchContext

        mandatory = kwargs.get("mandatory", [])

        # Carry forward contexts if present
        incoming_ctx: Optional[SearchContext] = kwargs.get("search_context")
        multi_day_ctx: Optional[Any] = kwargs.get("multi_day_context")

        # Ensure we have a ledger even if Phase 1 was skipped (e.g., unit tests)
        if incoming_ctx is None:
            incoming_ctx = SearchContext.initialize()

        # 1. Validate
        early_result = self._validate_mandatory(mandatory)
        if early_result is not None:
            return early_result[0], early_result[1], 0.0, incoming_ctx, multi_day_ctx

        # 2. Extract context
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")

        # 3. Load parameters
        capacity, revenue, cost_unit, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)
        # 4. Create subset problem
        # use_all_bins=True (VRPP mode) allows the solver to CONSIDER all bins,
        # but only forcefully visit the mandatory list (mandatory_nodes).
        sub_dist_matrix, sub_wastes, subset_indices, mandatory_nodes = self._create_subset_problem(
            mandatory,
            distance_matrix,
            bins,
            use_all_bins=current_expand_pool(),
        )

        # Build coordinate arrays aligned to the sub-problem node indices.
        coords = kwargs.get("coords")
        sub_node_ids = subset_indices[1:]
        x_coords: Optional[np.ndarray] = None
        y_coords: Optional[np.ndarray] = None

        if coords is not None and len(coords) > 0:
            try:
                # Depot at index 0: use the mean of all bin coordinates as a
                # reasonable depot proxy if the depot has no explicit entry,
                # or index into coords directly if the depot row is present.
                n = len(sub_node_ids)
                x_arr = np.zeros(n + 1)
                y_arr = np.zeros(n + 1)

                # Depot (index 0): use centroid or explicit depot coords
                depot_row = coords[coords["ID"] == 0]
                if len(depot_row) > 0:
                    x_arr[0] = float(depot_row["Lng"].iloc[0])
                    y_arr[0] = float(depot_row["Lat"].iloc[0])
                else:
                    x_arr[0] = float(coords["Lng"].mean())
                    y_arr[0] = float(coords["Lat"].mean())

                coord_index = coords.set_index("ID")
                for local_idx, original_id in enumerate(sub_node_ids, start=1):
                    if original_id in coord_index.index:
                        x_arr[local_idx] = float(coord_index.loc[original_id, "Lng"])
                        y_arr[local_idx] = float(coord_index.loc[original_id, "Lat"])
                    else:
                        # Missing coordinate: disable pruning for safety
                        x_arr = None  # type: ignore[assignment]
                        y_arr = None  # type: ignore[assignment]
                        break

                if x_arr is not None:
                    x_coords = x_arr
                    y_coords = y_arr
            except Exception:
                # Coordinate extraction is best-effort; pruning degrades gracefully to
                # no-op if anything goes wrong
                x_coords = None
                y_coords = None

        # 5. Run solver (subclass-specific)
        routes, profit, _ = self._run_solver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes,
            subset_indices=subset_indices,
            x_coords=x_coords,
            y_coords=y_coords,
            **kwargs,
        )

        # 6. Map back to global IDs
        tour = self._map_tour_to_global(routes, subset_indices)

        # 7. Compute cost
        cost = self._compute_cost(distance_matrix, tour)

        # 8. Merge ConstructionMetrics into the incoming SearchContext
        construction_patch: ConstructionMetrics = {
            "algorithm": type(self).__name__,
            "n_mandatory": len(mandatory),
            "profit": profit,
        }
        if incoming_ctx is not None:
            from logic.src.interfaces.context.search_context import SearchPhase

            out_ctx: Optional[SearchContext] = merge_context(
                incoming_ctx,
                phase=SearchPhase.ROUTING,
                construction_metrics=construction_patch,
            )
        else:
            out_ctx = None

        return tour, cost, profit, out_ctx, multi_day_ctx

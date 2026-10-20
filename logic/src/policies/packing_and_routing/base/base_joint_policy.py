"""Base classes for joint packing and routing policies.

Provides :class:`BaseJointPolicy`, an abstract base class for solvers that
perform mandatory-bin selection **and** routing in a single
integrated optimisation loop.

Attributes:
    BaseJointPolicy: The abstract base class defined in this module.

Example:
    >>> class MyJointPolicy(BaseJointPolicy):
    ...     def solve_joint(self, context):
    ...         return [], [], 0.0, 0.0
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.interfaces.context.joint_context import JointSelectionConstructionContext
from logic.src.interfaces.context.search_context import (
    ConstructionMetrics,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)
from logic.src.interfaces.routing_algorithm import IRoutingAlgorithm
from logic.src.tracking.viz_mixin import PolicyVizMixin


class BaseJointPolicy(PolicyVizMixin, IRoutingAlgorithm):
    """Abstract base class for joint packing and routing policies.

    These policies jointly handle packing and routing
    across a multi-day horizon, typically balancing immediate routing costs
    with long-term profit opportunities.

    Attributes:
        config (Optional[Any]): Configuration parameters for the policy.
    """

    def __init__(self, config: Any = None):
        """Initialize policy with optional config.

        Args:
            config: Configuration parameters. Defaults to None.
        """
        self._config = config
        self._seed = getattr(config, "seed", 42) if config is not None else 42

    @property
    def config(self) -> Any:
        """Return the policy configuration.

        Returns:
            Any: The policy configuration object.
        """
        return self._config

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the dataclass type for config parsing.

        Returns:
            Optional[Type]: The configuration class type.
        """
        return None

    def _get_config_key(self) -> str:
        """Return the Hydra configuration key.

        Default implementation uses the lowercase class name without 'policy'.

        Returns:
            str: The configuration key string.
        """
        return type(self).__name__.lower().replace("policy", "")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def solve_joint(
        self,
        context: JointSelectionConstructionContext,
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """Solve mandatory-packing and routing jointly.

        Args:
            context: The search and problem context.

        Returns:
            Tuple[List[int], List[List[int]], float, float]:
                - selected_bins: List of bin IDs selected for service.
                - routes: Constructed routes as lists of node indices.
                - profit: Total expected profit.
                - cost: Total routing cost.
        """

    # ------------------------------------------------------------------
    # IRoutingAlgorithm implementation
    # ------------------------------------------------------------------

    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[Any]]:
        """Simulation-engine entry point. Extracts context and calls solve_joint.

        Args:
            kwargs: Simulation context and parameters.

        Returns:
            Tuple[List[int], float, float, Optional[SearchContext], Optional[Any]]:
                - tour: Flat global-ID tour.
                - cost: Total routing cost.
                - profit: Total expected profit.
                - out_ctx: Updated search context.
                - multi_day_ctx: Multi-day state context.

        Raises:
            ValueError: If 'bins' or 'distance_matrix' are not provided in kwargs.
        """
        # 1. Extract inputs from kwargs (sim context)
        bins = kwargs.get("bins")
        distance_matrix = kwargs.get("distance_matrix")
        search_ctx = kwargs.get("search_context")
        multi_day_ctx = kwargs.get("multi_day_context")

        # Load area-specific constants if available
        from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

        area = kwargs.get("area", "generic")
        waste_type = kwargs.get("waste_type", "generic")
        # get_area_params returns (capacity_pct, revenue, density, expenses, volume).
        # JointSelectionConstructionContext.capacity is kilograms.
        capacity_pct, revenue_kg, bin_density, cost_per_km, bin_volume = load_area_and_waste_type_params(
            area, waste_type
        )
        capacity_kg = float(capacity_pct) / 100.0 * float(bin_volume) * float(bin_density)

        # 2. Build Joint Context
        if bins is not None and distance_matrix is not None:
            # Convert bins object (state) to arrays
            bin_ids = bins.bin_ids if hasattr(bins, "bin_ids") else np.arange(1, len(bins.c) + 1, dtype=np.int32)
            current_fill = bins.c if hasattr(bins, "c") else np.zeros(len(bin_ids))

            context = JointSelectionConstructionContext(
                bin_ids=bin_ids,
                current_fill=current_fill.copy(),
                distance_matrix=distance_matrix,
                capacity=capacity_kg,
                revenue_kg=float(revenue_kg),
                cost_per_km=float(cost_per_km),
                bin_density=float(bin_density),
                bin_volume=float(bin_volume),
                max_fill=100.0,
                n_vehicles=int(kwargs.get("n_vehicles", 1)),
                scenario_tree=kwargs.get("scenario_tree"),
                mandatory_override=kwargs.get("mandatory"),
            )
        else:
            raise ValueError("Missing 'bins' or 'distance_matrix' in execute kwargs.")

        # 3. Call core solver
        selected_bins, routes, profit, cost = self.solve_joint(context)

        # 4. Convert outcomes to 5-tuple
        if routes:
            _, _, subset_indices = self._build_subset_problem(selected_bins, context)
            tour = self._routes_to_flat_tour(routes, subset_indices)
        else:
            tour = [0, 0]

        # Metrics enrichment
        sel_patch: SelectionMetrics = {
            "strategy": type(self).__name__,
            "n_selected": len(selected_bins),
        }
        con_patch: ConstructionMetrics = {
            "algorithm": type(self).__name__,
            "n_mandatory": len(selected_bins),
            "profit": profit,
        }

        if search_ctx is not None:
            out_ctx: Optional[SearchContext] = merge_context(
                search_ctx,
                phase=SearchPhase.ROUTING,
                selection_metrics=sel_patch,
                construction_metrics=con_patch,
            )
        else:
            out_ctx = SearchContext.initialize(
                selection_metrics=sel_patch,
                metadata={"joint_solver": type(self).__name__},
            )
            out_ctx = merge_context(
                out_ctx,
                phase=SearchPhase.ROUTING,
                construction_metrics=con_patch,
            )

        return tour, cost, profit, out_ctx, multi_day_ctx

    # ------------------------------------------------------------------
    # Shared utility helpers
    # ------------------------------------------------------------------

    def _build_subset_problem(
        self,
        selected_bins: List[int],
        context: JointSelectionConstructionContext,
    ) -> Tuple[np.ndarray, Dict[int, float], List[int]]:
        """Build distance sub-matrix and wastes dict.

        Args:
            selected_bins: IDs of selected bins.
            context: Context containing problem data.

        Returns:
            Tuple[np.ndarray, Dict[int, float], List[int]]:
                - sub_dist: Distance sub-matrix for selected bins.
                - sub_wastes: Waste demands for selected bins.
                - subset_indices: Mapping from local index to global ID.
        """
        subset_indices = [0] + list(selected_bins)
        dist_np = np.asarray(context.distance_matrix, dtype=float)
        sub_dist = dist_np[np.ix_(subset_indices, subset_indices)]

        sub_wastes: Dict[int, float] = {}
        for local_i, global_i in enumerate(subset_indices[1:], start=1):
            sub_wastes[local_i] = float(context.current_fill[global_i - 1])

        return sub_dist, sub_wastes, subset_indices

    @staticmethod
    def _routes_to_flat_tour(
        routes: List[List[int]],
        subset_indices: List[int],
    ) -> List[int]:
        """Convert local-index routes to a flat global-ID tour.

        Args:
            routes: Routes as node indices.
            subset_indices: Local to global ID mapping.

        Returns:
            List[int]: Flat global-ID tour including depots.
        """
        tour: List[int] = [0]
        for route in routes:
            for node in route:
                tour.append(subset_indices[node])
            tour.append(0)
        if len(tour) == 1:
            tour = [0, 0]
        return tour

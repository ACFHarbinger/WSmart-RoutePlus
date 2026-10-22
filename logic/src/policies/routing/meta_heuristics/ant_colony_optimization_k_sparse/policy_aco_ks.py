r"""ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.

Attributes:
    ACOPolicy: K-Sparse Ant Colony Optimization policy class.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse import ACOPolicy
    >>> policy = ACOPolicy()
"""

import gc
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .solver import KSparseACOSolver


@RoutingAlgorithmRegistry.register("aco_ks")
class ACOPolicy(BaseRoutingPolicy):
    """K-Sparse Ant Colony Optimization policy class.

    Uses ACS with sparse pheromone matrix for efficient VRP solving.

    Attributes:
        config: Configuration for the policy.

    Example:
        >>> policy = ACOPolicy()
    """

    def __init__(self, config: Optional[Union[KSparseACOConfig, Dict[str, Any]]] = None):
        """Initialize ACO policy with optional config.

        Args:
            config: KSparseACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for ACO.

        Returns:
            The KSparseACOConfig class or None.
        """
        return KSparseACOConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the ACO policy.

        Returns:
            str: The registry key 'aco'.
        """
        return "aco"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute the K-Sparse Ant Colony Optimization (ACO) solver logic.

        This implementation uses the Ant Colony System (ACS) framework with a
        sparse pheromone matrix (K-Sparse). Ants construct solutions by
        probabilistically choosing next nodes based on pheromone levels and
        greedy desirability (profit-to-distance ratio). Pheromones are
        updated globally by the best-found solution (and optionally elitist
        ants) and locally by every ant during construction to encourage
        exploration.

        Args:
            sub_dist_matrix: Square distance matrix.
            sub_wastes: Node fill levels.
            capacity: Vehicle capacity.
            revenue: Revenue per kg.
            cost_unit: Cost per km.
            values: Merged config dictionary.
            mandatory_nodes: Nodes that must be visited.
            x_coords: Optional[np.ndarray] = None,
            y_coords: Optional[np.ndarray] = None,
            kwargs: Additional context.

        Returns:
            Tuple containing (routes, profit, cost).
        """
        # 1. Standardize Parameters
        params = KSparseACOConfig.from_config(self.config)

        # 2. THE MEMORY GUILLOTINE
        # Explicitly delete the O(n^2) distance matrix and force garbage collection
        # to ensure the low-memory environment constraints are met.
        if sub_dist_matrix is not None:
            del sub_dist_matrix
            gc.collect()

        # 3. Coordinate Projection and Preparation
        # We must combine x and y into a single (N, 2) array for the cKDTree.
        if x_coords is None or y_coords is None:
            raise ValueError(
                "K-Sparse ACO strictly requires 'x_coords' and 'y_coords' to "
                "reconstruct spatial data after freeing the distance matrix."
            )

        # Project lat/lng to a local equirectangular plane centered on the depot (index 0).
        # This prevents distortion in the k-nearest neighbor (KNN) selection.
        depot_lat = float(y_coords[0])
        depot_lng = float(x_coords[0])
        cos_lat = math.cos(math.radians(depot_lat))

        # O(n) projection
        x_proj = (x_coords - depot_lng) * cos_lat
        y_proj = y_coords - depot_lat

        # Stack into (N, 2) array for the solver's spatial index
        node_coords = np.column_stack((x_proj, y_proj))

        # 4. Execute K-Sparse Solver
        # The solver will build a cKDTree to compute distances and nearest
        # neighbors on-demand in O(n log n) time.
        solver = KSparseACOSolver(
            node_coords=node_coords,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        return solver.solve()

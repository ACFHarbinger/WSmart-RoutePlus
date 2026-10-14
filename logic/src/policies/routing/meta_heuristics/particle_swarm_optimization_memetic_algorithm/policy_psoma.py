"""
PSOMA Policy Adapter.

Adapts the Particle Swarm Optimization Memetic Algorithm (PSOMA) to the
agnostic BaseRoutingPolicy interface.

Attributes:
    PSOMAConfig (Type): Configuration schema for the PSOMA solver.
    BaseRoutingPolicy (Type): Abstract base for routing policies.
    RoutingAlgorithmRegistry (Type): Global registry for routing algorithms.

Example:
    >>> from logic.src.configs.policies.psoma import PSOMAConfig
    >>> config = PSOMAConfig(pop_size=20)
    >>> policy = PSOMAPolicy(config)
    >>> routes = policy.solve(problem)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.psoma import PSOMAConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .params import PSOMAParams
from .solver import PSOMASolver


@RoutingAlgorithmRegistry.register("psoma")
class PSOMAPolicy(BaseRoutingPolicy):
    """
    PSOMA policy class.

    Visits bins using Particle Swarm Optimization with a memetic local-search step.

    Attributes:
        solver (PSOMASolver): Internal solver instance.
        params (PSOMAParams): Algorithm parameters.
    """

    def __init__(self, config: Optional[Union[PSOMAConfig, Dict[str, Any]]] = None):
        """Initializes the PSOMA policy.

        Args:
            config (Optional[Union[PSOMAConfig, Dict[str, Any]]]): Configuration
                source for the Particle Swarm Optimization Memetic Algorithm.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for PSOMA.

        Returns:
            Optional[Type]: The PSOMAConfig class.
        """
        return PSOMAConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the PSOMA policy.

        Returns:
            str: The registry key 'psoma'.
        """
        return "psoma"

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
        Execute the Particle Swarm Optimization Memetic Algorithm (PSOMA)
        solver logic.

        PSOMA is a hybrid metaheuristic that combines the collective intelligence
        of Particle Swarm Optimization (PSO) with the individual refinement
        capabilities of a Memetic Algorithm. While the swarm particles explore
        the global profit surface through velocity-driven movements (cognitive
        and social attraction), individual particles are periodically refined
        using local search operators. This "education" step ensures that
        particles converge onto meaningful local optima, effectively bridging
        long-range exploration with rigorous intensification.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                PSOMA parameters (pop_size, omega, c1, c2, local_search_freq).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            kwargs (Any): Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = PSOMAParams.from_config(values)

        solver = PSOMASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()

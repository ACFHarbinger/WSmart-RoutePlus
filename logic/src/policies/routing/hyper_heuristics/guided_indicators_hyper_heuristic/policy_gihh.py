"""
GIHH Policy Adapter.

Adapts the Hyper-Heuristic with Two Guidance Indicators (GIHH) logic
to the common policy interface.

Attributes:
    GIHHPolicy: Hyper-Heuristic with Two Guidance Indicators policy class.

Examples:
    >>> from logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic import GIHHPolicy
    >>> policy = GIHHPolicy()
    >>> policy.solve()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gihh import GIHHConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.registry import ALL

from .gihh import GIHHSolver


@ALL.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("gihh")
class GIHHPolicy(BaseRoutingPolicy):
    """
    Hyper-Heuristic with Two Guidance Indicators policy class.

    Uses IRI (Improvement Rate Indicator) and TBI (Time-based Indicator)
    to adaptively select low-level heuristics during search.

    Attributes:
        config: GIHHConfig dataclass, raw dict from YAML, or None.
    """

    def __init__(self, config: Optional[Union[GIHHConfig, Dict[str, Any]]] = None):
        """Initialize GIHH policy with optional config.

        Args:
            config: GIHHConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return config class for GIHH.

        Returns:
            Type: Config class for GIHH.
        """
        return GIHHConfig

    def _get_config_key(self) -> str:
        """Return config key for GIHH.

        Returns:
            str: Config key for GIHH.
        """
        return "gihh"

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
        Execute the Guided Indicators Hyper-Heuristic (GIHH) solver logic.

        GIHH employs a multi-indicator selection mechanism (IRI and TBI) to
        dynamically choose between low-level heuristics. It maintains a Pareto
        archive (ARCH) of non-dominated solutions discovered during the search.

        This adapter performs an a posteriori selection from the final archive
        by choosing the solution with the highest total net profit for maximum
        compatibility with the simulation loop.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                hyperparameters like `alpha`, `beta`, `gamma`, etc.
            mandatory_nodes: Local indices of bins that MUST be
                collected in this period.
            kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = GIHHConfig(
            time_limit=float(values.get("time_limit", 60.0)),
            max_iterations=int(values.get("max_iterations", 1000)),
            seed=values.get("seed", 42),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seg=values.get("seg", 80),
            alpha=values.get("alpha", 0.5),
            beta=values.get("beta", 0.4),
            gamma=values.get("gamma", 0.1),
            min_prob=values.get("min_prob", 0.05),
            nonimp_threshold=values.get("nonimp_threshold", 150),
        )

        solver = GIHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        # GIHH returns the full Pareto archive (ARCH)
        arch = solver.solve()
        if not arch:
            return [], 0.0, 0.0

        # Select solution with highest scalar profit from the archive
        best_sol = max(arch, key=lambda s: s.profit)

        # Calculate final cost correctly using the solver's internal logic
        final_cost = solver._cost(best_sol.routes)
        return best_sol.routes, best_sol.profit, final_cost

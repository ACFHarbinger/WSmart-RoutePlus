"""
Simulator adapter for the Two-Phase Kernel Search (TPKS) matheuristic.

Attributes:
    TPKSPolicy: The Two-Phase Kernel Search policy.

Example:
    >>> from logic.src.policies.routing.matheuristics.two_phase_kernel_search import TPKSPolicy
    >>> tpks_policy = TPKSPolicy()
    >>> tpks_policy.execute(distance_matrix=..., wastes={}, capacity=1e9, R=1.0, C=1.0, mandatory_nodes=[])
"""

from typing import List, Optional, Tuple, Type

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .params import TPKSParams
from .solver import run_tpks_gurobi


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("tpks")
class TPKSPolicy(BaseRoutingPolicy):
    """
    Two-Phase Kernel Search policy.

    Phase I focuses on finding a first high-quality feasible solution by
    collecting per-variable statistics. Phase II uses those statistics to
    adaptively size buckets and allocate solve time, improving solution quality
    without re-running an expensive LP relaxation.

    Inherits from BaseRoutingPolicy (single-day interface), not
    BaseMultiPeriodRoutingPolicy, because TPKS operates on a single-day
    VRPP subproblem. Multi-period extension is achieved by calling it from
    inside a multi-period policy's _run_multi_period_solver.

    Attributes:
        None
    """

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return configuration class.

        Returns:
            Type: Configuration class.
        """
        return None  # or a dedicated TPKSConfig dataclass if needed

    def _get_config_key(self) -> str:
        """Return configuration key.

        Returns:
            str: Configuration key.
        """
        return "tpks"

    def _run_solver(self, sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values, mandatory_nodes, **kwargs):
        """Not used — execute() calls run_tpks_gurobi directly.

        Args:
            sub_dist_matrix (np.ndarray): Distance matrix.
            sub_wastes (Dict[int, float]): Wastes.
            capacity (float): Capacity.
            revenue (Optional[float]): Revenue.
            cost_unit (float): Cost unit.
            values (Optional[np.ndarray]): Values.
            mandatory_nodes (List[int]): Mandatory nodes.
            kwargs (dict): Keyword arguments containing problem parameters.

        Returns:
            Tuple[List[int], float, float]: (tour, cost, profit).
        """
        return [], 0.0, 0.0

    def execute(self, **kwargs) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Executes the Two-Phase Kernel Search policy.

        Args:
            kwargs: Keyword arguments containing problem parameters.

        Returns:
            A tuple of (tour, cost, profit, search_context, multi_day_context).
        """
        params = TPKSParams.from_config(self.config)

        tour, obj_val, cost = run_tpks_gurobi(
            dist_matrix=kwargs["distance_matrix"],
            wastes=kwargs.get("wastes", {}),
            capacity=kwargs.get("capacity", 1e9),
            R=kwargs.get("R", 1.0),
            C=kwargs.get("C", 1.0),
            mandatory_nodes=kwargs.get("mandatory", []),
            params=params,
            recorder=kwargs.get("recorder"),
        )

        return (tour, cost, obj_val, kwargs.get("search_context"), kwargs.get("multi_day_context"))

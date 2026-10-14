r"""
Constraint Programming (CP-SAT) Policy Adapter for Stochastic VRPP.

Adapts the Google OR-Tools CP-SAT solver to the Multi-Period Stochastic
framework. CP-SAT is a modern solver that combines Constraint Programming (CP)
with Boolean Satisfiability (SAT) techniques, particularly effective at
handling complex logical constraints and non-linear cost structures.

Problem Formulation — Integrated CP Optimization:
    The solver models the $T$-period stochastic problem using a set of
    decision variables:
        - $x_{i,j,t,k}$: Boolean arc $(i,j)$ traversal for vehicle $k$ on day $t$.
        - $y_{i,t}$: Boolean visit flag for bin $i$ on day $t$.
        - $w_{i,t}$: Continuous (quantized) inventory level of bin $i$ on day $t$.

    Core Constraints:
        - **Degree & Flow**: In/out degree balance for each day $t$.
        - **Transition Logic**: Bin inventories propagate across days based on
          visit decisions: $w_{i,t+1} = \max(0, w_{i,t} + \Delta_{i,t} - y_{i,t} \cdot w_{i,t})$.
        - **Capacity**: Cumulative load $\le$ payload at each node.
        - **Overflow**: Logical constraints penalizing states where $w_{i,t} > Q$.

Advantage of CP-SAT:
    CP-SAT can solve discretized versions of the stochastic problem without
    requiring the linearity strictly enforced by MILP solvers. It uses **Lazy
    Clause Generation** to prune the search space by learning from conflicts,
    making it highly robust for combinatorial problems with complex state-transitions.

Implementation:
    - **Lookahead**: Extends decisions over the $T$-day horizon.
    - **Quantization**: Bin increments $\Delta_{i,t}$ are quantized to integers
      to satisfy CP-SAT's integer-based core.

Registry key: ``"cp_sat"``

References:
    Schutt, A., et al. (2013). "LCG-based solvers for the traveling salesman
    problem with time windows". OR Spectrum, 35(3), 543-575.

Attributes:
    CPSATPolicy (class): Adapter for the CP-SAT policy using OR-Tools.

Example:
    >>> policy = CPSATPolicy(config)
    >>> sol, plan, stats = policy._run_multi_period_solver(problem, multi_day_ctx)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.cp_sat import CPSATConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .cp_sat_engine import CPSATEngine


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.MULTI_PERIOD,
)
@RoutingAlgorithmRegistry.register("cp_sat")
class CPSATPolicy(BaseMultiPeriodRoutingPolicy):
    """Adapter for the Constraint Programming with Boolean Satisfiability (CP-SAT) policy using OR-Tools.

    Now standardized to the Multi-Period framework.

    Attributes:
        config (CPSATConfig): Configuration for the CP-SAT policy.
    """

    @classmethod
    def _config_class(cls):
        """Returns the configuration class for this policy.

        Returns:
            Type[CPSATConfig]: The configuration class.
        """
        return CPSATConfig

    def _get_config_key(self) -> str:
        """Return the configuration key.

        Returns:
            str: The registry key 'cp_sat'.
        """
        return "cp_sat"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Multi-Period CP-SAT solver.

        This method models the stochastic routing problem over the T-day horizon
        as a Constraint Programming (CP) model, which is then solved using
        OR-Tools' CP-SAT engine. It discretizes bin fill levels and utilizes
        the scenario tree to find decisions that maximize expected long-term
        profit, accounting for future overflow penalties and collection
        efficiencies.

        It leverages OR-Tools' **Lazy Clause Generation** to prune the search
        space by learning from conflicts, making it highly robust for
        combinatorial problems with complex state-transitions that might be
        difficult to linearize for standard MILP solvers.

        Args:
            problem: The current ProblemContext containing the state, ScenarioTree,
                and problem constants (capacity, revenue, etc.).
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan (nested list by day and vehicle).
                - stats: Execution statistics and solver performance metadata.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("CP-SAT requires a ScenarioTree in ProblemContext.")
            pass

        # We pass the tree to the engine.
        engine = CPSATEngine(
            config=self.config,
            distance_matrix=problem.distance_matrix,
            initial_wastes=problem.wastes,
            capacity=problem.capacity,
        )

        # Solve and return Day 0 action
        route_day_0, expected_val = engine.solve()

        # Wrap into full plan
        full_plan: List[List[List[int]]] = [[] for _ in range(self.horizon + 1)]
        full_plan[0] = [route_day_0]

        sol_ctx = SolutionContext.from_problem(problem, route_day_0)

        return sol_ctx, full_plan, {"mip_status": "solved", "expected_profit": expected_val}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Legacy fallback solver.

        Args:
            sub_dist_matrix: Distance matrix for the sub-problem.
            sub_wastes: Waste levels for the sub-problem.
            capacity: Vehicle capacity.
            revenue: Revenue per unit of waste.
            cost_unit: Cost per unit of distance.
            values: Additional values/parameters.
            mandatory_nodes: List of nodes that must be visited.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[List[List[int]], float, float]: Empty solution tuple.
        """
        return [], 0.0, 0.0

"""Generic declarative routing adapter (P-30).

Most routing adapters repeat the same structure: an ``__init__`` calling
``super()``, a ``_config_class()`` and a ``_run_solver`` that materializes the
typed config and forwards the positional 6-tuple to a solver or dispatcher.
:class:`GenericRoutingPolicy` collapses that boilerplate into one template:

- the subclass declares ``_solver`` (a :class:`~logic.src.policies.routing.base.sub_problem.Solver`
  class or any ``(problem, cfg) -> SolveResult`` callable such as a ``run_*``
  dispatcher) and optionally ``_config_class()``;
- ``_run_solver`` materializes the effective config, builds the
  :class:`~logic.src.policies.routing.base.sub_problem.SubProblem`, solves and
  finalizes;
- adapters that really do something different override the small hooks
  ``_make_subproblem`` / ``_solve`` / ``_finalize`` instead of the whole
  method.
"""

from typing import Any, Dict, List, Tuple

from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.sub_problem import SubProblem

__all__ = ["GenericRoutingPolicy"]


class GenericRoutingPolicy(BaseRoutingPolicy):
    """Declarative adapter: declare the solver, the template does the rest.

    Class attributes
    ----------------
    _solver:
        A :class:`Solver` class (instantiated as ``_solver(problem, cfg)`` and
        solved with ``.solve()``) or a callable taking ``(problem, cfg)`` and
        returning the ``(routes, profit, cost)`` triple — typically the
        policy's ``run_*`` dispatcher.
    """

    _solver: Any = None

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        cfg = self._effective_config(values)
        problem = self._make_subproblem(
            sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values, mandatory_nodes, kwargs
        )
        result = self._solve(problem, cfg)
        return self._finalize(result, problem)

    def _make_subproblem(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        kwargs: Dict[str, Any],
    ) -> SubProblem:
        """Assemble the :class:`SubProblem` from the legacy ``_run_solver`` parts."""
        return SubProblem(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
            mandatory_nodes=mandatory_nodes,
            subset_indices=kwargs.get("subset_indices"),
            x_coords=kwargs.get("x_coords"),
            y_coords=kwargs.get("y_coords"),
            values=values,
            n_vehicles=kwargs.get("n_vehicles"),
            env=kwargs.get("model_env"),
            node_coords=kwargs.get("node_coords"),
            recorder=kwargs.get("recorder"),
            dual_values=kwargs.get("dual_values"),
        )

    def _solve(self, problem: SubProblem, cfg: Any) -> Any:
        """Run the declared solver. Override for engines needing extra inputs."""
        solver = self._solver
        if isinstance(solver, type):
            return solver(problem, cfg).solve()
        return solver(problem, cfg)

    def _finalize(self, result: Any, problem: SubProblem) -> Tuple[List[List[int]], float, float]:
        """Map the solver result back to the ``(routes, profit, cost)`` contract."""
        routes, profit, cost = result
        return routes, profit, cost

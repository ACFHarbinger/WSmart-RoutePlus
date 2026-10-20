"""Gurobi MIP master for logic-based Benders decomposition, accumulating no-good, optimality, Pareto and combinatorial cuts."""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    Optional,
    Set,
)

import gurobipy as gp
from gurobipy import (
    GRB,
    quicksum,
)

# ---------------------------------------------------------------------------
# LBBD master problem
# ---------------------------------------------------------------------------


class _LBBDMaster:
    """Gurobi MIP master that accumulates Benders cuts.

    Decision variables:
        y_i ∈ {0,1}  — whether customer i is visited.
        θ  ≥ 0       — routing-cost proxy (receives Benders cuts).

    Objective: max  Σ R·w_i·y_i  −  θ

    Attributes:
        n_nodes: Number of nodes.
        wastes: {local node → fill_level}.
        R: Revenue per unit waste.
        mandatory: Local indices of mandatory customers.
        env: Gurobi environment.
        seed: Random seed.
        mdl: Gurobi model.
        y: Decision variables for visit.
        theta: Routing cost proxy.
        _big_m: Big M constant.
        _cut_count: Number of cuts added.
    """

    def __init__(
        self,
        n_nodes: int,
        wastes: Dict[int, float],
        R: float,
        mandatory: Set[int],
        min_cover_ratio: float,
        vehicle_limit: Optional[int],
        env: Any,
        seed: int,
    ) -> None:
        """
        Args:
            n_nodes: Number of nodes.
            wastes: {local node → fill_level}.
            R: Revenue per unit waste.
            mandatory: Local indices of mandatory customers.
            min_cover_ratio: Minimum coverage ratio.
            vehicle_limit: Maximum number of vehicles.
            env: Gurobi environment.
            seed: Random seed.

        """
        self.n_nodes = n_nodes
        self.wastes = wastes
        self.R = R
        self.mandatory = mandatory
        self.env = env

        self.mdl = gp.Model("LBBD_Master", env=env) if env else gp.Model("LBBD_Master")
        self.mdl.Params.LogToConsole = 0
        self.mdl.Params.OutputFlag = 0
        self.mdl.Params.Seed = seed

        # y_i: visit binary
        self.y = self.mdl.addVars(range(1, n_nodes + 1), vtype=GRB.BINARY, name="y")
        # θ: routing cost proxy
        self.theta = self.mdl.addVar(lb=0.0, name="theta")

        # Mandatory coverage
        for i in mandatory:
            self.mdl.addConstr(self.y[i] == 1, name=f"mand_{i}")

        # Minimum coverage ratio
        if min_cover_ratio < 1.0:
            self.mdl.addConstr(
                quicksum(self.y[i] for i in range(1, n_nodes + 1)) >= math.ceil(min_cover_ratio * n_nodes),
                name="min_cover",
            )

        revenue_expr = quicksum(R * wastes.get(i, 0.0) * self.y[i] for i in range(1, n_nodes + 1))
        self.mdl.setObjective(revenue_expr - self.theta, GRB.MAXIMIZE)
        self.mdl.update()

        self._big_m = R * sum(wastes.values()) + 1.0
        self._cut_count = 0

    def solve(self, time_limit: float) -> Optional[Set[int]]:
        """Solve master and return selected node set, or None if infeasible.

        Args:
            time_limit: Time limit.

        Returns:
            Set of selected nodes.
        """
        self.mdl.Params.TimeLimit = max(1.0, time_limit)
        self.mdl.optimize()
        if self.mdl.SolCount == 0:
            return None
        return {i for i in range(1, self.n_nodes + 1) if self.y[i].X > 0.5}

    def get_lp_bound(self) -> float:
        """LP relaxation upper bound after last solve.

        Args:
            None

        Returns:
            LP relaxation upper bound.
        """
        try:
            return float(self.mdl.ObjBound)
        except Exception:
            return float("inf")

    def add_nogood_cut(self, Y: Set[int]) -> None:
        """Forbid the exact selection Y.

        Args:
            Y: Set of nodes.

        """
        expr = quicksum(self.y[i] for i in Y)
        self.mdl.addConstr(expr <= len(Y) - 1, name=f"nogood_{self._cut_count}")
        self._cut_count += 1
        self.mdl.update()

    def add_optimality_cut(self, Y: Set[int], sub_cost: float) -> None:
        """Add: θ ≥ sub_cost − M · (|Y| − Σ_{i∈Y} y_i).

        Args:
            Y: Set of nodes.
            sub_cost: Subproblem cost.

        """
        lhs = quicksum(self.y[i] for i in Y)
        self.mdl.addConstr(
            self.theta >= sub_cost - self._big_m * (len(Y) - lhs),
            name=f"opt_{self._cut_count}",
        )
        self._cut_count += 1
        self.mdl.update()

    def add_pareto_cut(self, Y: Set[int], sub_cost: float, primal_bound: float) -> None:
        """Magnanti-Wang Pareto-optimal strengthening of the optimality cut.

        Args:
            Y: Set of nodes.
            sub_cost: Subproblem cost.
            primal_bound: Primal bound.

        """
        gap = sub_cost - primal_bound
        if gap <= 1e-8:
            return
        lhs = quicksum(self.y[i] for i in Y)
        self.mdl.addConstr(
            self.theta >= gap * lhs - gap * len(Y) + primal_bound,
            name=f"pareto_{self._cut_count}",
        )
        self._cut_count += 1
        self.mdl.update()

    def add_combinatorial_cut(self, Y: Set[int], sub_cost: float) -> None:
        """Coverage-strengthening cut: every subset of Y needs the same routing.

        Args:
            Y: Set of nodes.
            sub_cost: Subproblem cost.

        """
        # For any T ⊆ Y, cost(T) ≥ cost(Y) − Σ_{i∉T, i∈Y} R·w_i
        # Approximated as: θ ≥ sub_cost − Σ_{i∈Y} R·w_i · (1 − y_i)
        penalty = quicksum(self.R * self.wastes.get(i, 0.0) * (1 - self.y[i]) for i in Y)
        self.mdl.addConstr(
            self.theta >= sub_cost - penalty,
            name=f"comb_{self._cut_count}",
        )
        self._cut_count += 1
        self.mdl.update()

    @property
    def n_cuts(self) -> int:
        """Return the number of cuts added to the master problem.

        Returns:
            Number of cuts added.
        """
        return self._cut_count

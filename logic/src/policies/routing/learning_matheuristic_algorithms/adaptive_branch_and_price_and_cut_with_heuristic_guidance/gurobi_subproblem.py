"""
Gurobi LP Subproblem for Temporal Benders Cuts.

Solves the LP relaxation of a single-vehicle VRPP for one (day, scenario)
pair given fixed master assignment decisions z̄. Returns the LP optimal
profit and dual multipliers on the assignment constraints for Benders cut
generation.

Attributes:
    GurobiVRPSubproblem: A class that solves the LP relaxation of a single-vehicle VRPP for Benders cut generation.

Example:
    >>> from gurobi_vrp_subproblem import GurobiVRPSubproblem
    >>> solver = GurobiVRPSubproblem()
    >>> profit, duals, route = solver.solve(z_bar_day, prizes, dist_matrix, loads)
"""

import logging
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

logger = logging.getLogger(__name__)


class GurobiVRPSubproblem:
    """
    LP relaxation of the single-vehicle VRPP for Benders cut generation.

    Formulation
    -----------
    For a given day d, scenario ξ, and master assignment z̄, the subproblem
    selects a set of bins to visit and a route through them:

    Variables (continuous relaxation):

        x[i, j] ∈ [0, 1]   arc (i → j) usage fraction
        y[i]    ∈ [0, 1]   fractional visit to bin i

    Objective:

        max  Σ_i π_i · y[i]  −  c · Σ_{i,j}  d_{ij} · x[i, j]

    where π_i is the scenario-augmented prize from ``ScenarioPrizeEngine``
    and c is ``cost_per_unit``.

    Constraints:

        (1) Assignment:  y[i] ≤ z̄[i]                    ∀i   ← duals μ_i
        (2) Out-flow:    Σ_j x[i,j]  =  y[i]             ∀i ≠ depot
        (3) In-flow:     Σ_j x[j,i]  =  y[i]             ∀i ≠ depot
        (4) Depot out:   Σ_j x[depot, j] ≤ 1
        (5) Depot in:    Σ_j x[j, depot] ≤ 1
        (6) Capacity:    Σ_i load_i · y[i] ≤ Q           (if loads provided)

    Note on relaxation quality
    --------------------------
    The LP omits subtour-elimination constraints (SECs), making it a valid
    but potentially loose relaxation of the true VRP.  Specifically,
    Q_LP(z̄, ξ) ≥ Q_VRP(z̄, ξ), so the generated Benders cut is valid for
    the LP-Benders relaxation but may be optimistic for the full MIP.

    A tighter bound can be obtained by solving a MIP with MTZ constraints
    (set ``relax=False`` in ``solve()``).  When ``relax=False`` the model is
    solved as a MIP, the integer solution is fixed, and the LP is re-solved
    at the optimal integer point to recover valid dual multipliers for the
    assignment constraints.

    Dual Recovery
    -------------
    The dual variable μ_i on constraint (1) is the shadow price of making
    bin i available to the router.  These form the Benders cut coefficients:

        θ_d  ≥  Q_LP(z̄)  +  Σ_i  μ_i · (z_{id} − z̄_{id})

    Optimisation over eligible nodes only
    --------------------------------------
    Arc and visit variables are created only for the eligible subset
    ``{i : z̄[i] > 0}``, reducing the LP size significantly when few bins
    are assigned on a given day.

    Attributes:
    ----------
    n_bins       : Total number of bins (matches master formulation).
    capacity     : Vehicle capacity Q used in the capacity constraint.
    cost_per_unit: Routing cost coefficient c (cost per unit of distance).
    time_limit   : Gurobi time limit per subproblem solve (seconds).
    """

    def __init__(
        self,
        n_bins: int,
        capacity: float,
        cost_per_unit: float,
        time_limit: float = 30.0,
    ) -> None:
        """
        Args:
            n_bins: Number of customer bins; must match the master formulation.
            capacity: Vehicle capacity Q.  Used in the optional capacity
                constraint Σ_i load_i · y[i] ≤ Q.  If loads are not provided
                at solve time, the capacity constraint is omitted.
            cost_per_unit: Cost coefficient c multiplied by arc distance;
                represents the cost per unit of distance traveled.
            time_limit: Maximum Gurobi solve time per subproblem in seconds.
                Applies both to the LP relaxation and the optional MIP solve.
        """
        self.n_bins = n_bins
        self.capacity = capacity
        self.cost_per_unit = cost_per_unit
        self.time_limit = time_limit

    # ------------------------------------------------------------------ #
    # Main Solve                                                            #
    # ------------------------------------------------------------------ #

    def solve(
        self,
        z_bar_day: Dict[int, int],
        prizes: Dict[int, float],
        dist_matrix: np.ndarray,
        loads: Optional[np.ndarray] = None,
        relax: bool = True,
    ) -> Tuple[float, Dict[int, float], List[int]]:
        """
        Solve the subproblem for one (day, scenario) pair.

        Args:
            z_bar_day: ``{bin_id: 0_or_1}`` assignment from the master for
                this specific day.  Only bins with value 1 are eligible.
            prizes: ``{bin_id: π_i}`` scenario-augmented node prizes from
                ``ScenarioPrizeEngine.scenario_weighted_prizes`` or per-scenario
                prizes from ``ScenarioPrizeEngine.compute_prizes``.
            dist_matrix: Distance matrix of shape ``(n_bins+1, n_bins+1)``.
                Index 0 is the depot; indices 1..n_bins are the bins.
                The subproblem only uses rows/columns for eligible nodes.
            loads: Optional ``np.ndarray`` of shape ``(n_bins,)`` where
                ``loads[i-1]`` is the fill level (load) of bin i.  Used in
                the capacity constraint.  Omit (``None``) to skip capacity.
            relax: If ``True`` (default), solve the LP relaxation and extract
                duals directly.  If ``False``, solve as a MIP with MTZ
                subtour elimination, fix the integer solution, and re-solve
                as an LP to recover duals on assignment constraints.

        Returns:
            profit: Subproblem optimal value Q(z̄, ξ).
            duals: ``{bin_id: μ_i}`` dual values on assignment constraints.
                Positive μ_i means making bin i available increases profit.
            route: Approximate feasible route ``[0, n_1, …, n_k, 0]``
                reconstructed from the LP/MIP solution via nearest-neighbour
                ordering.
        """
        eligible = [i for i in range(1, self.n_bins + 1) if z_bar_day.get(i, 0) > 0.5]

        if not eligible:
            logger.debug("Subproblem: no eligible bins; trivially empty.")
            return 0.0, {}, [0]

        if relax:
            return self._solve_lp(z_bar_day, prizes, dist_matrix, loads, eligible)
        else:
            return self._solve_mip_with_dual_recovery(z_bar_day, prizes, dist_matrix, loads, eligible)

    # ------------------------------------------------------------------ #
    # LP Relaxation                                                         #
    # ------------------------------------------------------------------ #

    def _solve_lp(
        self,
        z_bar_day: Dict[int, int],
        prizes: Dict[int, float],
        dist_matrix: np.ndarray,
        loads: Optional[np.ndarray],
        eligible: List[int],
    ) -> Tuple[float, Dict[int, float], List[int]]:
        """LP relaxation solve with direct dual extraction.

        Args:
            z_bar_day: Integer lower bound on bin selections.
            prizes: Bin prizes.
            dist_matrix: Travel cost matrix.
            loads: Bin loads.
            eligible: List of eligible bin indices.

        Returns:
            profit: Subproblem optimal value Q(z̄, ξ).
            duals: ``{bin_id: μ_i}`` dual values on assignment constraints.
                Positive μ_i means making bin i available increases profit.
            route: Approximate feasible route ``[0, n_1, …, n_k, 0]``
                reconstructed from the LP/MIP solution via nearest-neighbour
                ordering.
        """
        sub_nodes = [0] + eligible  # depot (0) + eligible bins

        model = gp.Model("VRP_Sub_LP")
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = self.time_limit
        model.Params.Method = 1  # Dual simplex — most reliable for LP duals

        arcs = [(i, j) for i in sub_nodes for j in sub_nodes if i != j]

        # Arc variables (continuous relaxation)
        x = model.addVars(arcs, lb=0.0, ub=1.0, name="x")

        # Visit variables (continuous relaxation)
        y = model.addVars(eligible, lb=0.0, ub=1.0, name="y")

        model.update()

        # Objective: max prize revenue − routing cost
        prize_terms = gp.quicksum(prizes.get(i, 0.0) * y[i] for i in eligible)
        cost_terms = self.cost_per_unit * gp.quicksum(float(dist_matrix[i, j]) * x[i, j] for i, j in arcs)
        model.setObjective(prize_terms - cost_terms, GRB.MAXIMIZE)

        # (1) Assignment constraints — source of Benders cut duals
        assign_constrs: Dict[int, gp.Constr] = {
            i: model.addConstr(y[i] <= float(z_bar_day.get(i, 0)), name=f"assign_{i}") for i in eligible
        }

        # (2) Out-flow conservation
        for i in eligible:
            model.addConstr(
                gp.quicksum(x[i, j] for j in sub_nodes if j != i) == y[i],
                name=f"out_{i}",
            )

        # (3) In-flow conservation
        for i in eligible:
            model.addConstr(
                gp.quicksum(x[j, i] for j in sub_nodes if j != i) == y[i],
                name=f"in_{i}",
            )

        # (4) Depot: at most one vehicle departs
        model.addConstr(
            gp.quicksum(x[0, j] for j in eligible) <= 1.0,
            name="depot_out",
        )

        # (5) Depot: at most one vehicle returns
        model.addConstr(
            gp.quicksum(x[j, 0] for j in eligible) <= 1.0,
            name="depot_in",
        )

        # (6) Capacity (if loads provided)
        if loads is not None:
            model.addConstr(
                gp.quicksum(float(loads[i - 1]) * y[i] for i in eligible) <= self.capacity,
                name="capacity",
            )

        model.optimize()

        profit, duals, route = 0.0, {}, [0]

        if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            profit = float(model.ObjVal)

            # Extract dual μ_i on assignment constraints
            for i, constr in assign_constrs.items():
                mu = float(constr.Pi)
                if abs(mu) > 1e-10:
                    duals[i] = mu

            # Reconstruct route from rounded LP solution
            visited = [i for i in eligible if y[i].X > 0.5]
            route = self._nearest_neighbour_route(visited, dist_matrix)

        elif model.Status == GRB.INFEASIBLE:
            logger.warning("VRP LP subproblem infeasible — check formulation.")
        else:
            logger.debug("VRP LP subproblem status: %d", model.Status)

        model.dispose()
        return profit, duals, route

    # ------------------------------------------------------------------ #
    # MIP + Dual Recovery                                                  #
    # ------------------------------------------------------------------ #

    def _solve_mip_with_dual_recovery(
        self,
        z_bar_day: Dict[int, int],
        prizes: Dict[int, float],
        dist_matrix: np.ndarray,
        loads: Optional[np.ndarray],
        eligible: List[int],
    ) -> Tuple[float, Dict[int, float], List[int]]:
        """
        Solve the VRPP as a MIP with MTZ subtour elimination, then fix the
        integer solution and re-solve as an LP to recover valid dual variables
        on the assignment constraints.

        This gives tighter Benders cuts than the LP relaxation at the cost of
        an additional LP solve per subproblem call.

        Args:
            z_bar_day: Integer lower bound on bin selections.
            prizes: Bin prizes.
            dist_matrix: Travel cost matrix.
            loads: Bin loads.
            eligible: List of eligible bin indices.

        Returns:
            profit: Subproblem optimal value Q(z̄, ξ).
            duals: {bin_id: μ_i} dual values on assignment constraints.
            route: Approximate feasible route [0, n_1, …, n_k, 0] reconstructed
                from the LP/MIP solution via nearest-neighbour ordering.
        """
        sub_nodes = [0] + eligible
        n_sub = len(eligible)
        arcs = [(i, j) for i in sub_nodes for j in sub_nodes if i != j]

        # ── Step 1: MIP solve ─────────────────────────────────────────
        mip = gp.Model("VRP_Sub_MIP")
        mip.Params.OutputFlag = 0
        mip.Params.LogToConsole = 0
        mip.Params.TimeLimit = self.time_limit
        mip.Params.MIPGap = 1e-4

        x_mip = mip.addVars(arcs, vtype=GRB.BINARY, name="x")
        y_mip = mip.addVars(eligible, vtype=GRB.BINARY, name="y")

        # MTZ position variables for subtour elimination
        # u[i] ∈ [1, n_sub] encodes the position of node i in the tour.
        # Constraint: u[i] - u[j] + n_sub * x[i,j] ≤ n_sub - 1  for i,j ≠ depot.
        u_mip = mip.addVars(eligible, lb=1.0, ub=float(n_sub), name="u")

        mip.update()

        # Objective
        prize_obj = gp.quicksum(prizes.get(i, 0.0) * y_mip[i] for i in eligible)
        cost_obj = self.cost_per_unit * gp.quicksum(float(dist_matrix[i, j]) * x_mip[i, j] for i, j in arcs)
        mip.setObjective(prize_obj - cost_obj, GRB.MAXIMIZE)

        # Assignment (z̄ upper bound)
        for i in eligible:
            mip.addConstr(y_mip[i] <= float(z_bar_day.get(i, 0)), name=f"assign_{i}")

        # Flow conservation
        for i in eligible:
            mip.addConstr(
                gp.quicksum(x_mip[i, j] for j in sub_nodes if j != i) == y_mip[i],
                name=f"out_{i}",
            )
            mip.addConstr(
                gp.quicksum(x_mip[j, i] for j in sub_nodes if j != i) == y_mip[i],
                name=f"in_{i}",
            )

        # Depot
        mip.addConstr(gp.quicksum(x_mip[0, j] for j in eligible) <= 1, name="depot_out")
        mip.addConstr(gp.quicksum(x_mip[j, 0] for j in eligible) <= 1, name="depot_in")

        # Capacity
        if loads is not None:
            mip.addConstr(
                gp.quicksum(float(loads[i - 1]) * y_mip[i] for i in eligible) <= self.capacity,
                name="capacity",
            )

        # MTZ subtour elimination: u[i] - u[j] + n * x[i,j] ≤ n-1  ∀(i,j), i≠depot
        for i in eligible:
            for j in eligible:
                if i != j:
                    mip.addConstr(
                        u_mip[i] - u_mip[j] + n_sub * x_mip[i, j] <= n_sub - 1,
                        name=f"mtz_{i}_{j}",
                    )

        mip.optimize()

        if mip.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or mip.SolCount == 0:
            logger.debug(
                "VRP MIP subproblem status: %d; falling back to LP relaxation.",
                mip.Status,
            )
            mip.dispose()
            return self._solve_lp(z_bar_day, prizes, dist_matrix, loads, eligible)

        mip_profit = float(mip.ObjVal)
        y_int = {i: round(y_mip[i].X) for i in eligible}
        visited_int = [i for i in eligible if y_int[i] > 0.5]
        route = self._nearest_neighbour_route(visited_int, dist_matrix)
        mip.dispose()

        # ── Step 2: Fix integer solution, re-solve as LP for duals ────
        lp_fix = gp.Model("VRP_Sub_LP_Fix")
        lp_fix.Params.OutputFlag = 0
        lp_fix.Params.LogToConsole = 0
        lp_fix.Params.Method = 1  # Dual simplex

        y_lp = lp_fix.addVars(eligible, lb=0.0, ub=1.0, name="y")
        lp_fix.update()

        # Fix visit decisions to the MIP-optimal solution
        for i in eligible:
            lp_fix.addConstr(y_lp[i] == float(y_int[i]), name=f"fix_y_{i}")

        # Assignment constraints (these are the ones we want duals for)
        assign_constrs_lp: Dict[int, gp.Constr] = {
            i: lp_fix.addConstr(y_lp[i] <= float(z_bar_day.get(i, 0)), name=f"assign_{i}") for i in eligible
        }

        lp_fix.setObjective(
            gp.quicksum(prizes.get(i, 0.0) * y_lp[i] for i in eligible),
            GRB.MAXIMIZE,
        )

        lp_fix.optimize()

        duals: Dict[int, float] = {}
        if lp_fix.Status == GRB.OPTIMAL:
            for i, constr in assign_constrs_lp.items():
                mu = float(constr.Pi)
                if abs(mu) > 1e-10:
                    duals[i] = mu

        lp_fix.dispose()
        return mip_profit, duals, route

    # ------------------------------------------------------------------ #
    # Route Reconstruction                                                 #
    # ------------------------------------------------------------------ #

    def _nearest_neighbour_route(self, visited: List[int], dist_matrix: np.ndarray) -> List[int]:
        """
        Build a feasible tour over ``visited`` using nearest-neighbour
        insertion starting from the depot (node 0).

        This is a O(n²) greedy heuristic — sufficient for the route
        extraction step where solution quality is secondary to feasibility.

        Args:
            visited: Unordered list of bin node ids to include.
            dist_matrix: Full distance matrix (depot at index 0).

        Returns:
            Route ``[0, n_1, …, n_k, 0]`` traversing all nodes in ``visited``.
        """
        if not visited:
            return [0]

        route = [0]
        remaining = list(visited)

        while remaining:
            last = route[-1]
            nearest = min(remaining, key=lambda n: float(dist_matrix[last, n]))
            route.append(nearest)
            remaining.remove(nearest)

        route.append(0)
        return route

"""
Gurobi Master MIP for Temporal Benders Decomposition.

The master problem coordinates the multi-day bin assignment decisions z_{id}
and provides an optimistic upper bound on the expected horizon profit via
surrogate variables θ_d that are progressively tightened by Benders optimality
cuts delivered from the routing subproblems.

Attributes:
    GurobiMasterProblem: A class that solves the master MIP of the Temporal Benders Decomposition.

Example:
    >>> from gurobi_master import GurobiMasterProblem
    >>> master = GurobiMasterProblem()
    >>> master.solve()
"""

import contextlib
import logging
from typing import Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


class GurobiMasterProblem:
    """
    Gurobi MIP master problem for the Temporal Benders loop.

    Formulation
    -----------
    Decision variables:

        z[i, d] ∈ {0, 1}   : 1 iff bin i is targeted for routing on day d.
        θ[d]    ≥ 0         : Surrogate for expected routing profit on day d.
                              Initialised with a user-supplied upper bound and
                              tightened by Benders optimality cuts.

    Objective:

        max  Σ_d  θ[d]

    Structural constraints:

        Σ_d z[i, d]  ≤  max_visits_per_bin    ∀i   (service frequency cap)

    Benders optimality cuts (added iteratively via add_benders_cut):

        θ[d]  ≥  constant  +  Σ_i  coeff[i] · z[i, d]    ∀ aggregated cut

    The initial upper bound on θ[d] prevents the master from being unbounded
    before the first cuts are added.  After the first round of subproblem
    solves the bound is progressively tightened toward the true optimum.

    Convergence Role
    ----------------
    The master objective value provides the Benders upper bound (UB).  UB
    monotonically decreases as cuts are added.  The algorithm stops when
    UB is within ``convergence_tol`` of the best primal lower bound (LB)
    computed from the subproblem objectives.

    Attributes:
    ----------
    n_bins    : Total number of bins (1-indexed).
    horizon   : Number of planning days T.
    z         : gurobipy.Vars dict, z[bin_id, day].
    theta     : gurobipy.Vars dict, theta[day].
    model     : The underlying gp.Model instance.
    """

    def __init__(
        self,
        n_bins: int,
        horizon: int,
        max_visits_per_bin: int = 1,
        theta_upper_bound: float = 1e6,
        mip_gap: float = 1e-4,
        time_limit: float = 60.0,
        output_flag: bool = False,
    ) -> None:
        """
        Args:
            n_bins: Number of customer bins (1-indexed, depot not counted).
            horizon: Planning horizon length T (number of days).
            max_visits_per_bin: Maximum times bin i may be assigned across all
                days: Σ_d z[i,d] ≤ max_visits_per_bin.  Use 1 (default) for
                the standard at-most-once-per-horizon assignment.
            theta_upper_bound: Initial upper bound on each θ[d] variable.
                Prevents the master from being unbounded before the first cuts
                are added.  Should exceed the true optimal daily profit; a safe
                choice is the sum of all node prizes for the highest-prize day.
            mip_gap: Relative MIP optimality gap tolerance for Gurobi.
            time_limit: Maximum Gurobi solve time in seconds per master call.
            output_flag: Set to True to enable Gurobi console output (useful
                for debugging; suppressed by default).

        Returns:
            None
        """
        self.n_bins = n_bins
        self.horizon = horizon
        self._cut_count = 0

        self._build(
            max_visits_per_bin,
            theta_upper_bound,
            mip_gap,
            time_limit,
            output_flag,
        )

    # ------------------------------------------------------------------ #
    # Model Construction                                                   #
    # ------------------------------------------------------------------ #

    def _build(
        self,
        max_visits: int,
        theta_ub: float,
        mip_gap: float,
        time_limit: float,
        output_flag: bool,
    ) -> None:
        """Construct the Gurobi model from scratch.

        Args:
            max_visits: Maximum times bin i may be assigned across all days.
            theta_ub: Initial upper bound on each theta[d] variable.
            mip_gap: Relative MIP optimality gap tolerance for Gurobi.
            time_limit: Maximum Gurobi solve time in seconds per master call.
            output_flag: Set to True to enable Gurobi console output.

        Returns:
            None
        """
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", int(output_flag))
        env.setParam("LogToConsole", int(output_flag))
        env.start()

        self.model = gp.Model("ABPC_HG_Master", env=env)
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit

        bins = range(1, self.n_bins + 1)
        days = range(self.horizon)

        # Binary assignment: z[bin_id, day]
        self.z = self.model.addVars(bins, days, vtype=GRB.BINARY, name="z")

        # Surrogate daily profit: theta[day] (upper-bounded to prevent
        # unboundedness before the first cuts are added)
        self.theta = self.model.addVars(
            days,
            lb=0.0,
            ub=theta_ub,
            vtype=GRB.CONTINUOUS,
            name="theta",
        )

        # Objective: max Σ_d θ[d]
        self.model.setObjective(
            gp.quicksum(self.theta[d] for d in days),
            GRB.MAXIMIZE,
        )

        # Service frequency: each bin visited at most max_visits times
        for i in bins:
            self.model.addConstr(
                gp.quicksum(self.z[i, d] for d in days) <= max_visits,
                name=f"freq_{i}",
            )

        self.model.update()

    # ------------------------------------------------------------------ #
    # Cut Interface                                                         #
    # ------------------------------------------------------------------ #

    def add_benders_cut(self, cut: Dict) -> None:
        """
        Add a single aggregated Benders optimality cut to the master model.

        Cut form (derived in ``TemporalBendersCoordinator.generate_benders_cut``):

            θ[d]  ≥  constant  +  Σ_i  coefficients[i] · z[i, d]

        The cut is immediately active: ``model.update()`` is called before
        returning.

        Args:
            cut: Dictionary with keys:
                ``"day"``          – day index d ∈ [0, T−1].
                ``"constant"``     – scalar RHS constant.
                ``"coefficients"`` – {bin_id: linear coefficient of z[bin_id, d]}.

        Returns:
            None
        """
        d = cut["day"]
        constant = float(cut["constant"])
        coeffs: Dict = cut.get("coefficients", {})

        if coeffs:
            expr = gp.quicksum(float(coeffs[i]) * self.z[i, d] for i in coeffs if 1 <= i <= self.n_bins)
            self.model.addConstr(
                self.theta[d] >= constant + expr,
                name=f"cut_{self._cut_count}",
            )
        else:
            # Constant-only cut: θ[d] ≥ constant (no variable contribution)
            self.model.addConstr(
                self.theta[d] >= constant,
                name=f"cut_{self._cut_count}",
            )

        self._cut_count += 1
        self.model.update()

    def add_benders_cuts_bulk(self, cuts: List[Dict]) -> None:
        """
        Add multiple cuts in a single ``model.update()`` call.

        Prefer this over repeated ``add_benders_cut`` calls when adding an
        entire iteration's worth of cuts (one per day) at once, as it reduces
        Gurobi model-update overhead.

        Args:
            cuts: List of cut dictionaries (same format as ``add_benders_cut``).

        Returns:
            None
        """
        for cut in cuts:
            d = cut["day"]
            constant = float(cut["constant"])
            coeffs: Dict = cut.get("coefficients", {})

            if coeffs:
                expr = gp.quicksum(float(coeffs[i]) * self.z[i, d] for i in coeffs if 1 <= i <= self.n_bins)
                self.model.addConstr(
                    self.theta[d] >= constant + expr,
                    name=f"cut_{self._cut_count}",
                )
            else:
                self.model.addConstr(
                    self.theta[d] >= constant,
                    name=f"cut_{self._cut_count}",
                )
            self._cut_count += 1

        self.model.update()

    # ------------------------------------------------------------------ #
    # Solve & Solution Extraction                                          #
    # ------------------------------------------------------------------ #

    def solve(self) -> Optional[Dict[int, Dict[int, int]]]:
        """
        Solve the master MIP and extract the bin-assignment solution z̄.

        Returns:
            z_bar: ``{day: {bin_id: 0_or_1}}`` on success.
            ``None`` if the model is infeasible or no feasible solution exists
            within the time limit.
        """
        self.model.optimize()
        status = self.model.Status

        if status == GRB.INFEASIBLE:
            logger.error("Master MIP is infeasible. Check that frequency constraints allow a feasible assignment.")
            return None

        if status == GRB.TIME_LIMIT:
            if self.model.SolCount == 0:
                logger.warning("Master MIP hit time limit with no feasible solution found.")
                return None
            logger.warning(
                "Master MIP hit time limit; returning best feasible solution (gap=%.2f%%).",
                100.0 * self.model.MIPGap,
            )

        elif status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("Master MIP terminated with unexpected status %d.", status)
            return None

        z_bar: Dict[int, Dict[int, int]] = {
            d: {i: (1 if self.z[i, d].X > 0.5 else 0) for i in range(1, self.n_bins + 1)} for d in range(self.horizon)
        }
        return z_bar

    # ------------------------------------------------------------------ #
    # Bound Accessors                                                       #
    # ------------------------------------------------------------------ #

    def get_objective_value(self) -> float:
        """
        Return the master objective value — the current Benders upper bound.

        Before any cuts this equals horizon × theta_upper_bound.  Each added
        cut strictly reduces this value toward the true optimum (assuming
        valid cuts).

        Returns ``+inf`` if the model has not been solved successfully.

        Returns:
            Objective value as a float (``float("inf")`` if not solved successfully).
        """
        try:
            return self.model.ObjVal
        except (AttributeError, gp.GurobiError):
            return float("inf")

    def get_mip_gap(self) -> float:
        """Return the relative MIP gap of the last solve (0.0 = optimal).

        Returns:
            MIP gap as a float (0.0 = optimal).
        """
        try:
            return self.model.MIPGap
        except (AttributeError, gp.GurobiError):
            return float("inf")

    # ------------------------------------------------------------------ #
    # Resource Management                                                  #
    # ------------------------------------------------------------------ #

    def dispose(self) -> None:
        """Free Gurobi model and associated environment resources.

        This releases the memory and other resources held by the Gurobi model.
        It is good practice to call this method when the master problem is no longer needed.

        Returns:
            None
        """
        with contextlib.suppress(Exception):
            self.model.dispose()

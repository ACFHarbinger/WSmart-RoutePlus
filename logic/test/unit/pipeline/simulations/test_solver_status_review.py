"""Review regressions for daily status isolation and forced-visit retries."""

import numpy as np
import pytest
from logic.src.pipeline.simulations.day_context import SimulationDayContext, run_day
from logic.src.pipeline.simulations.solver_status import current_solver_status, note_solver_status, reset_solver_status

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_fill_failure_does_not_inherit_previous_day_solver_status(monkeypatch):
    from logic.src.pipeline.simulations.actions import FillAction

    def fail(_self, _context):
        raise ValueError("bad daily fill")

    monkeypatch.setattr(FillAction, "execute", fail)
    ctx = SimulationDayContext(policy_name="alns", day=2)
    note_solver_status("gurobi:OPTIMAL")
    try:
        with pytest.raises(ValueError, match="bad daily fill"):
            run_day(ctx)
        assert ctx.daily_log["solver_status"] == "ValueError: bad daily fill"
    finally:
        reset_solver_status()


def test_forced_visit_retry_retains_both_solver_statuses():
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.ortools_wrapper import (
        _run_ortools_tcf_optimizer,
    )

    reset_solver_status()
    try:
        # Each bin fits, but both mandatory bins do not fit in a one-vehicle fleet.
        # The wrapper retries after relaxing *all* forcing constraints.
        route, _, _ = _run_ortools_tcf_optimizer(
            bins=np.array([100.0, 100.0]),
            distance_matrix=[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            values={"Omega": 0.0, "psi": 1.0, "Q": 100.0, "R": 1.0, "C": 1.0},
            binsids=[1, 2],
            mandatory_nodes=[1, 2],
            number_vehicles=1,
            solver_id="SCIP",
            time_limit=5,
        )
        assert len(set(route) - {0}) == 1
        assert current_solver_status() == "ortools:INFEASIBLE -> ortools:OPTIMAL"
    finally:
        reset_solver_status()

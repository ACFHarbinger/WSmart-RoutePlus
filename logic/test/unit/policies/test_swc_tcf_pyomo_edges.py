"""SWC-TCF pyomo backend edge cases (B-kimi-54)."""

import numpy as np
import pytest

pytest.importorskip("pyomo")
pytest.importorskip("gurobipy")

from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.dispatcher import (  # noqa: E402
    run_swc_tcf_optimizer,
)

pytestmark = [pytest.mark.unit]


def test_no_depot_arcs_returns_an_empty_day_instead_of_raising(capsys):
    """All depot arcs beyond MAX_ARC_DISTANCE_KM made pyomo reject `0 == 0` as a trivial Boolean."""
    dist = [[0, 7000, 7000], [7000, 0, 5], [7000, 5, 0]]
    values = {"Omega": 0.1, "psi": 1, "Q": 150.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
    try:
        with capsys.disabled():  # pyomo's own output capture clashes with pytest's
            route, _profit, _cost = run_swc_tcf_optimizer(
                bins=np.array([95.0, 95.0]),
                distance_matrix=dist,
                values=values,
                binsids=[0, 1, 2],
                mandatory_nodes=[1, 2],
                number_vehicles=1,
                time_limit=10,
                framework="pyomo",
                optimizer="gurobi",
            )
    except RuntimeError as exc:  # an explicit infeasibility report is acceptable; a pyomo ValueError is not
        assert "infeasible" in str(exc).lower()
        return
    assert route in ([0], [0, 0])

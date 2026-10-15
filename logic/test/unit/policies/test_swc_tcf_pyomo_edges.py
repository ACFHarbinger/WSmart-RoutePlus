"""SWC-TCF pyomo backend edge cases (B-kimi-54)."""

import numpy as np
import pytest

pytest.importorskip("pyomo")
pytest.importorskip("gurobipy")

from logic.src.configs.policies.swc_tcf import SWCTCFConfig
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.dispatcher import (  # noqa: E402
    run_swc_tcf_optimizer,
)

pytestmark = [pytest.mark.unit]


def test_no_depot_arcs_returns_an_empty_day_instead_of_raising(capsys):
    """All depot arcs beyond MAX_ARC_DISTANCE_KM made pyomo reject `0 == 0` as a trivial Boolean."""
    dist = [[0, 7000, 7000], [7000, 0, 5], [7000, 5, 0]]
    try:
        with capsys.disabled():  # pyomo's own output capture clashes with pytest's
            route, _profit, _cost = run_swc_tcf_optimizer(
                build_subproblem(
                    np.array(dist),
                    {1: 95.0, 2: 95.0},
                    150.0,
                    1.0,
                    1.0,
                    mandatory_nodes=[1, 2],
                    n_vehicles=1,
                    values={"Omega": 0.1, "psi": 1, "B": 1.0, "V": 1.0},
                ),
                SWCTCFConfig(framework="pyomo", engine="gurobi", time_limit=10),
            )
    except RuntimeError as exc:  # an explicit infeasibility report is acceptable; a pyomo ValueError is not
        assert "infeasible" in str(exc).lower()
        return
    assert route in ([0], [0, 0])

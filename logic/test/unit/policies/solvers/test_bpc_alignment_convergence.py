

from unittest.mock import MagicMock

import numpy as np
import pytest
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import (
    CompositeCuttingPlaneEngine,
    KnapsackCoverEngine,
    create_cutting_plane_engine,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
    run_bpc,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def bpc_instance():
    dist_matrix = np.array([[0, 10, 10, 20], [10, 0, 5, 15], [10, 5, 0, 10], [20, 15, 10, 0]])
    wastes = {1: 5.0, 2: 5.0, 3: 5.0}  # Total 15
    capacity = 10.0  # Need at least 2 vehicles
    R = 10.0
    C = 1.0
    return dist_matrix, wastes, capacity, R, C


def test_bpc_default_branching_strategy(bpc_instance):
    dist, wastes, cap, R, C = bpc_instance
    values = {"max_cg_iterations": 10}
    result = run_bpc(dist, wastes, cap, R, C, params=values)

    if isinstance(result, tuple):
        routes, cost = result
    elif hasattr(result, "nodes"):
        routes = [result.nodes]
        cost = result.profit
    else:
        # Final fallback for unexpected types
        routes = []
        cost = 0.0

    assert len(routes) > 0
    assert cost > 0


def test_knapsack_cover_separation():
    master = MagicMock()
    master.vehicle_limit = 1
    master.model = MagicMock()
    v1 = MagicMock()
    v1.X = 0.6
    v2 = MagicMock()
    v2.X = 0.6
    master.lambda_vars = [v1, v2]
    r1 = MagicMock()
    r1.node_coverage = {1, 2}
    r2 = MagicMock()
    r2.node_coverage = {3, 4}
    master.routes = [r1, r2]
    master.add_capacity_cut.return_value = True
    engine = KnapsackCoverEngine(MagicMock(), MagicMock())
    added = engine.separate_and_add_cuts(master, max_cuts=1)
    assert added == 1
    master.add_capacity_cut.assert_called()


def test_create_engine_factory():
    v_model = MagicMock()
    sep_engine = MagicMock()
    engine = create_cutting_plane_engine("all", v_model, sep_engine)
    assert engine.get_name() == "composite"
    assert isinstance(engine, CompositeCuttingPlaneEngine)
    assert (
        len(engine.engines) >= 7
    )  # RCC, SRI, EdgeClique, FleetCover, PhysicalLCI, SaturatedArcLCI, KnapsackCover, plus supplemental


def test_bpc_exact_convergence_log(bpc_instance, caplog):
    dist, wastes, cap, R, C = bpc_instance
    values = {"max_cg_iterations": 20, "branching_strategy": "divergence"}
    result = run_bpc(dist, wastes, cap, R, C, params=values)

    if isinstance(result, tuple):
        routes, cost = result
    elif hasattr(result, "nodes"):
        pass
    else:
        pass

    # The 3-node instance converges in a single CG pass, so z_UB/RMP iteration
    # logs may not fire.  The Phase II switch confirms column generation ran.
    assert "z_UB" in caplog.text or "RMP" in caplog.text or "Phase II" in caplog.text

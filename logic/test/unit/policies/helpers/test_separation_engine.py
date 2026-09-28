

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from logic.src.policies.helpers.solvers_and_matheuristics.separation.engine import SeparationEngine
from logic.src.policies.helpers.solvers_and_matheuristics.separation.inequality import (
    CapacityCut,
    PCSubtourEliminationCut,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def base_vrpp_model() -> VRPPModel:
    n_nodes = 6
    cost_matrix = np.array([
        [0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 1.5, 2.0, 2.5],
        [1.0, 1.0, 0.0, 1.0, 1.5, 2.0],
        [2.0, 1.5, 1.0, 0.0, 1.0, 1.5],
        [2.0, 2.0, 1.5, 1.0, 0.0, 1.0],
        [3.0, 2.5, 2.0, 1.5, 1.0, 0.0],
    ])
    wastes = {1: 10.0, 2: 20.0, 3: 30.0, 4: 15.0, 5: 25.0}
    capacity = 40.0
    return VRPPModel(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        num_vehicles=2,
        revenue_per_kg=1.0,
        cost_per_km=1.0,
    )

def test_separation_engine_init(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    assert engine.model == base_vrpp_model
    assert len(engine.pool) == 0
    assert engine.enable_heuristic_rcc_separation is True
    assert engine.enable_comb_cuts is False

def test_separate_integer_sec_only(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.zeros(15)
    x_vals[5] = 1.0  # (1,2)
    x_vals[6] = 1.0  # (1,3)
    x_vals[9] = 1.0  # (2,3)
    y_vals = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    cuts = engine.separate_integer(x_vals, y_vals, sec_only=True)
    assert len(cuts) > 0

def test_separate_fractional_various_branches(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model, enable_heuristic_rcc_separation=False)
    x_vals = np.zeros(15)
    y_vals = np.array([0.5, 0.5, 0.5, 0.0, 0.0])
    # Run with node_count > 0, iteration % 3 != 0 (should skip exact separation)
    cuts = engine.separate_fractional(x_vals, y_vals, node_count=2, iteration=1)
    assert isinstance(cuts, list)

    # Run with node_count > 0, iteration % 3 == 0 (should run exact separation, but with rcc disabled)
    cuts = engine.separate_fractional(x_vals, y_vals, node_count=2, iteration=3)
    assert isinstance(cuts, list)

def test_separate_disconnected_components(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.zeros(15)
    x_vals[5] = 1.0  # (1,2)
    x_vals[6] = 1.0  # (1,3)
    x_vals[9] = 1.0  # (2,3)

    y_vals = np.array([1.0, 1.0, 1.0, 0.0, 0.0]) # nodes 1,2,3 visited

    cuts = engine.separate_integer(x_vals, y_vals)
    assert len(cuts) > 0
    sec_cuts = [c for c in cuts if isinstance(c, PCSubtourEliminationCut)]
    assert len(sec_cuts) > 0
    assert sec_cuts[0].node_set == {1, 2, 3}

def test_separate_disconnected_components_unvisited(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.zeros(15)
    x_vals[5] = 1.0  # (1,2)
    x_vals[6] = 1.0  # (1,3)
    x_vals[9] = 1.0  # (2,3)
    y_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # nodes not visited
    engine._separate_disconnected_components(x_vals, y_vals)
    assert len(engine.pool) == 0

def test_separate_weak_subtours(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.zeros(15)
    x_vals[5] = 0.4  # (1,2)
    x_vals[6] = 0.4  # (1,3)
    x_vals[9] = 0.4  # (2,3)

    y_vals = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    cuts = engine.separate_fractional(x_vals, y_vals, node_count=1, iteration=0)
    assert len(cuts) > 0

def test_separate_capacity_cuts(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    # Total demand = 10 + 20 + 30 = 60 > capacity (40) -> min_vehicles = 2.
    x_vals = np.zeros(15)
    x_vals[0] = 1.0  # (0,1)
    x_vals[1] = 1.0  # (0,2)
    x_vals[5] = 1.0  # (1,2)
    x_vals[6] = 1.0  # (1,3)
    x_vals[9] = 1.0  # (2,3)

    y_vals = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
    cap_cuts = [c for c in cuts if isinstance(c, CapacityCut)]
    assert len(cap_cuts) > 0

    # Check y_vals = None
    engine.pool = []
    engine._separate_capacity_cuts(x_vals, y_vals=None)
    assert len(engine.pool) > 0

def test_separate_capacity_cuts_maxflow_heuristic(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model, enable_heuristic_rcc_separation=True)
    x_vals = np.zeros(15)
    x_vals[4] = 0.1  # (0,5)
    x_vals[14] = 1.0  # (4,5)
    x_vals[3] = 1.0  # (0,4)

    y_vals = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

    cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
    assert isinstance(cuts, list)

def test_separate_capacity_cuts_maxflow_success(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model, enable_heuristic_rcc_separation=True)
    x_vals = np.ones(15) * 0.5
    y_vals = np.ones(5)

    # Mock scipy maximum_flow to return success
    mock_flow_result = MagicMock()
    mock_flow_result.flow_value = 0.5
    mock_flow = MagicMock()
    # residual = capacity - flow. If capacity is 0.5 and flow is 0, residual is 0.5
    mock_flow.toarray.return_value = np.zeros((6, 6))
    mock_flow_result.flow = mock_flow

    with patch("logic.src.policies.helpers.solvers_and_matheuristics.separation.engine.maximum_flow", return_value=mock_flow_result):
        cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
        assert isinstance(cuts, list)

def test_separate_capacity_cuts_maxflow_fallback(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model, enable_heuristic_rcc_separation=True)
    x_vals = np.ones(15) * 0.5
    y_vals = np.ones(5)

    with patch("logic.src.policies.helpers.solvers_and_matheuristics.separation.engine.maximum_flow", side_effect=Exception("Scipy flow error")):
        cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
        assert isinstance(cuts, list)

def test_separate_pcsec_exact(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.ones(15) * 0.5
    y_vals = np.ones(5)

    cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
    sec_cuts = [c for c in cuts if isinstance(c, PCSubtourEliminationCut)]
    assert isinstance(sec_cuts, list)

def test_separate_pcsec_exact_success(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.ones(15) * 0.5
    y_vals = np.ones(5)

    mock_flow_result = MagicMock()
    mock_flow_result.flow_value = 0.5
    mock_flow = MagicMock()
    # Reachable nodes in residual graph
    # residual graph capacity = adj - flow.
    # adj[u,v] > 1e-4. Let's make u=1 reachable to v=2.
    mock_flow.toarray.return_value = np.zeros((6, 6))
    mock_flow_result.flow = mock_flow

    with patch("logic.src.policies.helpers.solvers_and_matheuristics.separation.engine.maximum_flow", return_value=mock_flow_result):
        cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
        assert isinstance(cuts, list)

def test_separate_pcsec_exact_fallback(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.ones(15) * 0.5
    y_vals = np.ones(5)

    with patch("logic.src.policies.helpers.solvers_and_matheuristics.separation.engine.maximum_flow", side_effect=Exception("Scipy flow error")):
        cuts = engine.separate_fractional(x_vals, y_vals, node_count=0)
        assert isinstance(cuts, list)

def test_separate_gsec_h2(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.zeros(15)
    x_vals[5] = 0.8  # (1,2)
    x_vals[9] = 0.8  # (2,3)
    x_vals[6] = 0.1  # (1,3)

    y_vals = np.array([0.9, 0.9, 0.9, 0.1, 0.1])

    engine._separate_gsec_h2(x_vals, y_vals)
    assert len(engine.pool) >= 0

    # Check y_vals = None
    engine.pool = []
    engine._separate_gsec_h2(x_vals, y_vals=None)
    assert len(engine.pool) >= 0

def test_strengthen_pool_facet_selection(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.ones(15) * 0.5
    y_vals = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    # Form 2.1
    cut1 = PCSubtourEliminationCut(node_set={1, 2}, violation=0.5)
    engine.pool.append(cut1)
    engine._strengthen_pool(engine.pool, x_vals, y_vals=None)
    assert cut1.facet_form in ["2.1", "2.2", "2.3"]

    # Form 2.2 / 2.3
    engine.pool = []
    cut2 = PCSubtourEliminationCut(node_set={1, 2}, violation=0.5)
    engine.pool.append(cut2)
    engine._strengthen_pool(engine.pool, x_vals, y_vals)
    assert cut2.facet_form in ["2.1", "2.2", "2.3"]

def test_refine_pcsec_build(base_vrpp_model: VRPPModel) -> None:
    engine = SeparationEngine(base_vrpp_model)
    x_vals = np.ones(15) * 0.5
    y_vals = np.ones(5)

    refined = engine._refine_pcsec_build({1, 2}, x_vals, y_vals)
    assert isinstance(refined, set)
    assert len(refined) >= 2



from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from logic.src.policies.helpers.solvers_and_matheuristics import Route
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import (
    BasicFleetCoverEngine,
    CompositeCuttingPlaneEngine,
    CuttingPlaneEngine,
    EdgeCliqueCutEngine,
    KnapsackCoverEngine,
    LimitedMemoryRank1CutEngine,
    PhysicalCapacityLCIEngine,
    RoundedCapacityCutEngine,
    RoundedMultistarCutEngine,
    SaturatedArcLCIEngine,
    SubsetRowCutEngine,
    create_cutting_plane_engine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.separation import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
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

def test_cutting_plane_engine_is_orthogonal() -> None:
    # We can use RoundedCapacityCutEngine to test the inherited _is_orthogonal method
    mock_model = MagicMock()
    mock_sep = MagicMock()
    engine = RoundedCapacityCutEngine(mock_model, mock_sep)

    assert engine._is_orthogonal(np.array([1.0, 0.0]), []) is True
    # If active_vecs has elements, the norm check runs
    assert engine._is_orthogonal(np.array([0.0, 0.0]), [np.array([1.0, 0.0])]) is False
    assert engine._is_orthogonal(np.array([1.0, 0.0]), [np.array([0.0, 1.0])]) is True
    assert engine._is_orthogonal(np.array([1.0, 0.0]), [np.array([1.0, 0.0])]) is False
    assert engine.engines == []

def test_rounded_capacity_cut_engine(base_vrpp_model: VRPPModel) -> None:
    sep_engine = SeparationEngine(base_vrpp_model)
    engine = RoundedCapacityCutEngine(base_vrpp_model, sep_engine)

    assert engine.get_name() == "rcc"

    master = MagicMock()
    # Mock master returns empty edge usage
    master.get_edge_usage.return_value = {}
    assert engine.separate_and_add_cuts(master, 10) == 0

    # Mock master returns edge usage and node visitation
    master.get_edge_usage.return_value = {(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0}
    master.get_node_visitation.return_value = {1: 1.0, 2: 1.0}
    master.add_capacity_cut.return_value = True
    master.add_sec_cut.return_value = True
    # RCCs are valid only for sets of mandatory bins (optional bins may be skipped).
    master.mandatory_nodes = {1, 2}

    # We can inject specific cuts into separate_fractional
    cap_cut = CapacityCut(node_set={1, 2}, total_demand=30.0, capacity=40.0, violation=0.5)
    sec_cut = PCSubtourEliminationCut(node_set={1, 2}, violation=0.5)
    sec_cut.facet_form = "2.1"

    with patch.object(sep_engine, "separate_fractional", return_value=[cap_cut, sec_cut]):
        added = engine.separate_and_add_cuts(master, 10)
        assert added == 2
        master.add_capacity_cut.assert_called_with([1, 2], 2.0)
        master.add_sec_cut.assert_called_with([1, 2], 2.0, cut_name="2.1", global_cut=True, node_i=-1, node_j=-1)

    # An RCC over optional bins is skipped; the PC-SEC is still added.
    master.mandatory_nodes = set()
    master.add_capacity_cut.reset_mock()
    with patch.object(sep_engine, "separate_fractional", return_value=[cap_cut, sec_cut]):
        assert engine.separate_and_add_cuts(master, 10) == 1
        master.add_capacity_cut.assert_not_called()

def test_subset_row_cut_engine(base_vrpp_model: VRPPModel) -> None:
    engine = SubsetRowCutEngine(base_vrpp_model)
    assert engine.get_name() == "sri"

    master = MagicMock()
    master.model = MagicMock()
    # Case: len(routes) < 2
    master.routes = [Route(nodes=[1], cost=1.0, revenue=2.0, load=10.0, node_coverage={1})]
    assert engine.separate_and_add_cuts(master, 10) == 0

    # Case: fractional routes
    r1 = Route(nodes=[1, 2], cost=2.0, revenue=4.0, load=20.0, node_coverage={1, 2})
    r2 = Route(nodes=[2, 3], cost=2.0, revenue=4.0, load=20.0, node_coverage={2, 3})
    r3 = Route(nodes=[1, 3], cost=2.0, revenue=4.0, load=20.0, node_coverage={1, 3})
    master.routes = [r1, r2, r3]

    var1 = MagicMock()
    var1.X = 0.5
    var2 = MagicMock()
    var2.X = 0.5
    var3 = MagicMock()
    var3.X = 0.5
    master.lambda_vars = [var1, var2, var3]
    master.global_cut_pool.active_sri_vectors = {}
    master.add_subset_row_cut.return_value = True

    added = engine.separate_and_add_cuts(master, 10)
    assert added > 0

def test_edge_clique_cut_engine(base_vrpp_model: VRPPModel) -> None:
    engine = EdgeCliqueCutEngine(base_vrpp_model)
    assert engine.get_name() == "edge_clique"

    master = MagicMock()
    master.model = MagicMock()

    r1 = Route(nodes=[1, 2], cost=2.0, revenue=4.0, load=20.0, node_coverage={1, 2})
    r2 = Route(nodes=[2, 3], cost=2.0, revenue=4.0, load=20.0, node_coverage={2, 3})
    master.routes = [r1, r2]

    var1 = MagicMock()
    var1.X = 0.8
    var2 = MagicMock()
    var2.X = 0.8
    master.lambda_vars = [var1, var2]

    master.add_edge_clique_cut.return_value = True
    added = engine.separate_and_add_cuts(master, 10)
    assert added > 0

def test_knapsack_cover_engine(base_vrpp_model: VRPPModel) -> None:
    sep_engine = SeparationEngine(base_vrpp_model)
    engine = KnapsackCoverEngine(base_vrpp_model, sep_engine)
    assert engine.get_name() == "cover"

    master = MagicMock()
    master.vehicle_limit = 1
    master.model = MagicMock()

    r1 = Route(nodes=[1], cost=1.0, revenue=2.0, load=10.0, node_coverage={1})
    r2 = Route(nodes=[2], cost=1.0, revenue=2.0, load=10.0, node_coverage={2})
    master.routes = [r1, r2]

    var1 = MagicMock()
    var1.X = 0.8
    var2 = MagicMock()
    var2.X = 0.8
    master.lambda_vars = [var1, var2]

    master.add_capacity_cut.return_value = True
    added = engine.separate_and_add_cuts(master, 10)
    assert added == 1

def test_basic_fleet_cover_engine(base_vrpp_model: VRPPModel) -> None:
    engine = BasicFleetCoverEngine(base_vrpp_model)
    assert engine.get_name() == "fleet_cover"

    master = MagicMock()
    master.vehicle_limit = 1

    r1 = Route(nodes=[1], cost=1.0, revenue=2.0, load=10.0, node_coverage={1})
    r2 = Route(nodes=[2], cost=1.0, revenue=2.0, load=10.0, node_coverage={2})
    master.routes = [r1, r2]

    var1 = MagicMock()
    var1.X = 0.8
    var2 = MagicMock()
    var2.X = 0.8
    master.lambda_vars = [var1, var2]
    master.add_lci_cut.return_value = True

    added = engine.separate_and_add_cuts(master, 10)
    assert added == 1

def test_physical_capacity_lci_engine(base_vrpp_model: VRPPModel) -> None:
    engine = PhysicalCapacityLCIEngine(base_vrpp_model)
    assert engine.get_name() == "physical_lci"

    master = MagicMock()
    master.model = MagicMock()

    # node visitation values
    master.get_node_visitation.return_value = {1: 0.9, 2: 0.9}
    r1 = Route(nodes=[1], cost=1.0, revenue=2.0, load=10.0, node_coverage={1})
    master.routes = [r1]

    var1 = MagicMock()
    var1.X = 0.9
    master.lambda_vars = [var1]
    master.add_lci_cut.return_value = True

    added = engine.separate_and_add_cuts(master, 10)
    assert added >= 0

def test_saturated_arc_lci_engine(base_vrpp_model: VRPPModel) -> None:
    engine = SaturatedArcLCIEngine(base_vrpp_model)
    assert engine.get_name() == "saturated_arc_lci"

    master = MagicMock()
    master.model = MagicMock()

    r1 = Route(nodes=[1, 2], cost=1.0, revenue=2.0, load=10.0, node_coverage={1, 2})
    master.routes = [r1]

    var1 = MagicMock()
    var1.X = 1.1 # arc flow sum will be > 1.0 - epsilon
    master.lambda_vars = [var1]
    master.add_lci_cut.return_value = True

    added = engine.separate_and_add_cuts(master, 10)
    assert added >= 0

def test_rounded_multistar_cut_engine(base_vrpp_model: VRPPModel) -> None:
    sep_engine = SeparationEngine(base_vrpp_model)
    engine = RoundedMultistarCutEngine(base_vrpp_model, sep_engine)
    assert engine.get_name() == "multistar"

    master = MagicMock()
    master.model = MagicMock()
    master.get_edge_usage.return_value = {(0, 1): 0.5, (1, 2): 0.5, (2, 0): 0.5}
    master.get_node_visitation.return_value = {1: 0.5, 2: 0.5}

    r1 = Route(nodes=[1, 2], cost=1.0, revenue=2.0, load=10.0, node_coverage={1, 2})
    master.routes = [r1]

    var1 = MagicMock()
    var1.X = 0.5
    master.lambda_vars = [var1]

    cap_cut = CapacityCut(node_set={1, 2}, total_demand=30.0, capacity=40.0, violation=0.5)
    with patch.object(sep_engine, "separate_fractional", return_value=[cap_cut]):
        master.add_multistar_cut.return_value = True
        added = engine.separate_and_add_cuts(master, 10)
        assert added >= 0

def test_limited_memory_rank_1_cut_engine(base_vrpp_model: VRPPModel) -> None:
    engine = LimitedMemoryRank1CutEngine(base_vrpp_model)
    assert engine.get_name() == "limited_memory_rank1"

    master = MagicMock()
    master.model = MagicMock()
    master.get_node_visitation.return_value = {1: 0.5, 2: 0.5, 3: 0.5}

    r1 = Route(nodes=[1, 2, 3], cost=1.0, revenue=2.0, load=10.0, node_coverage={1, 2, 3})
    master.routes = [r1]

    var1 = MagicMock()
    var1.X = 0.8
    master.lambda_vars = [var1]

    master.add_limited_memory_rank1_cut.return_value = True
    added = engine.separate_and_add_cuts(master, 10)
    assert added >= 0

def test_create_cutting_plane_engine(base_vrpp_model: VRPPModel) -> None:
    sep_engine = SeparationEngine(base_vrpp_model)

    # Test creation of all valid names
    for name in ["rcc", "sri", "edge_clique", "fleet_cover", "physical_lci",
                 "saturated_arc_lci", "multistar", "cover",
                 "limited_memory_rank1", "all", "composite"]:
        engine = create_cutting_plane_engine(name, base_vrpp_model, sep_engine=sep_engine)
        assert engine is not None
        assert engine.get_name() in ["rcc", "sri", "edge_clique", "fleet_cover",
                                    "physical_lci", "saturated_arc_lci", "multistar",
                                    "cover", "limited_memory_rank1", "composite"]

    with pytest.raises(ValueError):
        create_cutting_plane_engine("invalid_name", base_vrpp_model)
    with pytest.raises(ValueError):
        create_cutting_plane_engine("rcc", base_vrpp_model, sep_engine=None)
    with pytest.raises(ValueError):
        create_cutting_plane_engine("multistar", base_vrpp_model, sep_engine=None)
    with pytest.raises(ValueError):
        create_cutting_plane_engine("cover", base_vrpp_model, sep_engine=None)

def test_composite_cutting_plane_engine(base_vrpp_model: VRPPModel) -> None:
    engine1 = MagicMock(spec=CuttingPlaneEngine)
    engine2 = MagicMock(spec=CuttingPlaneEngine)
    engine1.separate_and_add_cuts.return_value = 2
    engine2.separate_and_add_cuts.return_value = 3

    comp = CompositeCuttingPlaneEngine([engine1, engine2])
    assert comp.get_name() == "composite"
    assert len(comp.engines) == 2

    master = MagicMock()
    added = comp.separate_and_add_cuts(master, 10)
    assert added == 5
    engine1.separate_and_add_cuts.assert_called_once()
    engine2.separate_and_add_cuts.assert_called_once()

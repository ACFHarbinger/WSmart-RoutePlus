

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from logic.src.configs.policies.routing.exact_and_decomposition_solvers.ils_bd import IntegerLShapedBendersConfig
from logic.src.policies.helpers.solvers_and_matheuristics import (
    CapacityCut,
    PCSubtourEliminationCut,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.routing.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem import (
    GUROBI_AVAILABLE,
    InventoryMasterProblem,
    MasterProblem,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


if GUROBI_AVAILABLE:
    from gurobipy import GRB
else:
    GRB = MagicMock()


def _has_gurobi_license() -> bool:
    try:
        import gurobipy as _gp



        _m = _gp.Model("_check")
        _m.dispose()
        return True
    except Exception:
        return False


_needs_license = pytest.mark.skipif(
    not _has_gurobi_license(),
    reason="Requires a valid Gurobi license",
)


@pytest.fixture
def vrpp_instance():
    dist_matrix = np.array([[0.0, 10.0, 15.0], [10.0, 0.0, 5.0], [15.0, 5.0, 0.0]])
    wastes = {1: 50.0, 2: 30.0}
    capacity = 100.0
    return dist_matrix, wastes, capacity


def test_master_problem_gurobi_missing():
    dist_matrix = np.zeros((2, 2))
    model = VRPPModel(n_nodes=2, cost_matrix=dist_matrix, wastes={}, capacity=100.0)
    params = IntegerLShapedBendersConfig()

    with (
        patch(
            "logic.src.policies.routing.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.GUROBI_AVAILABLE",
            False,
        ),
        pytest.raises(ImportError, match="Gurobi \\(gurobipy\\) is required"),
    ):
        MasterProblem(model, params)


@_needs_license
def test_master_problem_basic_flow(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=True)

    master = MasterProblem(model, params)

    # Test pre-build error
    with pytest.raises(RuntimeError, match="MasterProblem.build\\(\\) must be called before add_optimality_cut"):
        master.add_optimality_cut(1.0, {1: 2.0})

    master.build()

    # Verify variables created
    assert len(master._x_vars) == 3
    assert len(master._y_vars) == 2
    assert master._theta_var is not None

    # Test add optimality cut
    master.add_optimality_cut(10.0, {1: 2.0, 2: 3.0})
    assert master._benders_cut_count == 1
    assert master.stats["benders_cuts"] == 1.0

    # Solve
    routes, y_hat, theta_hat, obj = master.solve()
    assert isinstance(routes, list)
    assert isinstance(y_hat, dict)
    assert isinstance(theta_hat, float)
    assert isinstance(obj, float)


@_needs_license
def test_master_problem_solve_no_solution(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=False)

    master = MasterProblem(model, params)
    master.build()

    # Replace master._gurobi_model with a MagicMock to avoid C extension patching issues
    mock_model = MagicMock()
    mock_model.SolCount = 0
    mock_model.NodeCount = 12.0
    mock_model.Runtime = 1.5
    master._gurobi_model = mock_model

    routes, y_hat, theta_hat, obj = master.solve()
    assert routes == []
    assert y_hat == {1: 0.0, 2: 0.0}
    assert theta_hat == params.theta_lower_bound
    assert obj == float("inf")


@_needs_license
def test_lazy_constraint_callback_integer_cuts(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=False)

    master = MasterProblem(model, params)
    master.build()

    # Mock separator.separate_integer to return various cuts
    cut_2_1 = PCSubtourEliminationCut(node_set={1}, violation=0.1, facet_form="2.1")
    cut_2_2 = PCSubtourEliminationCut(node_set={1, 2}, violation=0.1, facet_form="2.2")
    cut_2_2.node_i = 1
    cut_2_3 = PCSubtourEliminationCut(node_set={1, 2}, violation=0.1, facet_form="2.3")
    cut_2_3.node_i = 1
    cut_2_3.node_j = 2
    cap_cut = CapacityCut(node_set={1, 2}, total_demand=150.0, capacity=100.0, violation=0.5)

    mock_model = MagicMock()
    mock_model.cbGetSolution.side_effect = lambda vars: [0.0] * len(vars)

    # 1. Test PC-SEC 2.1
    with patch.object(master.separator, "separate_integer", return_value=[cut_2_1]):
        master._add_integer_cuts(mock_model)
        mock_model.cbLazy.assert_called()
        assert master.stats["sec_cuts"] == 1.0

    # 2. Test PC-SEC 2.2
    mock_model.reset_mock()
    with patch.object(master.separator, "separate_integer", return_value=[cut_2_2]):
        master._add_integer_cuts(mock_model)
        mock_model.cbLazy.assert_called()
        assert master.stats["sec_cuts"] == 2.0

    # 3. Test PC-SEC 2.3
    mock_model.reset_mock()
    with patch.object(master.separator, "separate_integer", return_value=[cut_2_3]):
        master._add_integer_cuts(mock_model)
        mock_model.cbLazy.assert_called()
        assert master.stats["sec_cuts"] == 3.0

    # 4. Test CapacityCut
    mock_model.reset_mock()
    with patch.object(master.separator, "separate_integer", return_value=[cap_cut]):
        master._add_integer_cuts(mock_model)
        mock_model.cbLazy.assert_called()
        assert master.stats["capacity_cuts"] == 1.0


@_needs_license
def test_lazy_constraint_callback_fractional_cuts(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=False)

    master = MasterProblem(model, params)
    master.build()

    cut_2_1 = PCSubtourEliminationCut(node_set={1}, violation=0.1, facet_form="2.1")
    cut_2_2 = PCSubtourEliminationCut(node_set={1, 2}, violation=0.1, facet_form="2.2")
    cut_2_2.node_i = 1
    cut_2_3 = PCSubtourEliminationCut(node_set={1, 2}, violation=0.1, facet_form="2.3")
    cut_2_3.node_i = 1
    cut_2_3.node_j = 2
    cap_cut = CapacityCut(node_set={1, 2}, total_demand=150.0, capacity=100.0, violation=0.5)

    mock_model = MagicMock()
    mock_model.cbGetNodeRel.side_effect = lambda vars: [0.5] * len(vars)
    mock_model.cbGet.return_value = 10  # node count

    # 1. Test PC-SEC 2.1
    with patch.object(master.separator, "separate_fractional", return_value=[cut_2_1]):
        master._add_fractional_cuts(mock_model)
        mock_model.cbCut.assert_called()

    # 2. Test PC-SEC 2.2
    mock_model.reset_mock()
    with patch.object(master.separator, "separate_fractional", return_value=[cut_2_2]):
        master._add_fractional_cuts(mock_model)
        mock_model.cbCut.assert_called()

    # 3. Test PC-SEC 2.3
    mock_model.reset_mock()
    with patch.object(master.separator, "separate_fractional", return_value=[cut_2_3]):
        master._add_fractional_cuts(mock_model)
        mock_model.cbCut.assert_called()

    # 4. Test CapacityCut
    mock_model.reset_mock()
    with patch.object(master.separator, "separate_fractional", return_value=[cap_cut]):
        master._add_fractional_cuts(mock_model)
        mock_model.cbCut.assert_called()


@_needs_license
def test_lazy_constraint_callback_dispatcher(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=False)

    master = MasterProblem(model, params)
    master.build()

    mock_model = MagicMock()
    # Mock callback where == MIPSOL
    with patch.object(master, "_add_integer_cuts") as mock_add_int:
        master._lazy_constraint_callback(mock_model, GRB.Callback.MIPSOL)
        mock_add_int.assert_called_with(mock_model)

    # Mock callback where == MIPNODE and status == OPTIMAL
    mock_model.cbGet.return_value = GRB.OPTIMAL
    with patch.object(master, "_add_fractional_cuts") as mock_add_frac:
        master._lazy_constraint_callback(mock_model, GRB.Callback.MIPNODE)
        mock_add_frac.assert_called_with(mock_model)


@_needs_license
def test_inventory_master_problem(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=False)

    demand_matrix = np.array([[10.0, 20.0], [15.0, 25.0]])
    bin_capacities = np.array([100.0, 100.0])
    initial_inventory = np.array([10.0, 20.0])

    # Test pre-build error
    inv_master = InventoryMasterProblem(
        model,
        params,
        horizon=2,
        demand_matrix=demand_matrix,
        bin_capacities=bin_capacities,
        initial_inventory=initial_inventory,
    )
    with pytest.raises(RuntimeError, match="must be called before build_inventory"):
        inv_master.build_inventory()

    inv_master.build()
    inv_master.build_inventory()

    # Check variables created
    assert len(inv_master._I_vars) == 4  # 2 nodes * 2 days
    assert len(inv_master._q_vars) == 4
    assert len(inv_master._s_vars) == 4

    # Test solve and plan retrieval
    routes, y_hat, theta_hat, obj = inv_master.solve()
    assert isinstance(routes, list)

    inv_plan = inv_master.get_inventory_plan()
    col_plan = inv_master.get_collection_plan()

    assert isinstance(inv_plan, dict)
    assert isinstance(col_plan, dict)
    assert len(inv_plan) == 4
    assert len(col_plan) == 4


@_needs_license
def test_inventory_master_problem_no_solution(vrpp_instance):
    dist, wastes, cap = vrpp_instance
    model = VRPPModel(n_nodes=3, cost_matrix=dist, wastes=wastes, capacity=cap, num_vehicles=2)
    params = IntegerLShapedBendersConfig(verbose=False)

    inv_master = InventoryMasterProblem(model, params, horizon=2)
    inv_master.build()
    inv_master.build_inventory()

    inv_master._gurobi_model = MagicMock()
    inv_master._gurobi_model.SolCount = 0
    inv_master._gurobi_model.NodeCount = 5.0
    inv_master._gurobi_model.Runtime = 0.5

    assert inv_master.get_inventory_plan() == {}
    assert inv_master.get_collection_plan() == {}

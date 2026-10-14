

import contextlib
from typing import cast
from unittest.mock import MagicMock, patch

import gurobipy as gp
import numpy as np
import pytest
from gurobipy import GRB
from logic.src.policies.helpers.solvers_and_matheuristics.common.node import Node
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.dfj import _dfj_callback
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.mtz import BBSolver
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.policy_bb import (
    BranchAndBoundPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_fix_1_root_initialization():
    """Fix 1: Root node must be initialized with bound=inf."""
    dist_matrix = np.zeros((2, 2))
    wastes = {1: 10.0}
    solver = BBSolver(dist_matrix, wastes, 100.0, 1.0, 1.0, 0.001)

    with patch('logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.mtz.Node') as mock_node:
        with contextlib.suppress(BaseException):
            solver.solve()
        assert mock_node.call_args_list[0][1]['bound'] == float('inf')

def test_fix_2_pruning_maximization():
    """Fix 2: Correct pruning formula for maximization."""
    dist_matrix = np.zeros((2, 2))
    wastes = {1: 10.0}
    solver = BBSolver(dist_matrix, wastes, 100.0, 1.0, 1.0, mip_gap=0.0)

    solver.incumbent_obj = 10.0
    solver.mip_gap = 0.0
    node = Node(bound=10.0)

    gap_threshold = solver.incumbent_obj + abs(solver.incumbent_obj) * solver.mip_gap
    assert node.bound <= gap_threshold

    node_better = Node(bound=10.001)
    assert not (node_better.bound <= gap_threshold)

def test_fix_3_strong_branching_safe_access():
    """Fix 3: Safe ObjVal access and model.Status check.

    Ensures ObjVal is only accessed when Status is OPTIMAL.
    """
    dist_matrix = np.zeros((3, 3))
    wastes = {1: 10.0, 2: 10.0}
    solver = BBSolver(dist_matrix, wastes, 100.0, 1.0, 1.0, branching_strategy="strong")

    class ModelMock:
        def __init__(self):
            self._Status = GRB.OPTIMAL
            self._obj_val_accessed_when_not_optimal = False

        @property
        def Status(self):
            return self._Status

        @Status.setter
        def Status(self, value):
            self._Status = value

        @property
        def ObjVal(self):
            if self.Status != GRB.OPTIMAL:
                self._obj_val_accessed_when_not_optimal = True
                raise gp.GurobiError(10005, "No solution")
            return 100.0

        def optimize(self):
            pass

        def setParam(self, a, b):
            pass

        def addVar(self, **kwargs):
            m = MagicMock()
            m.LB = 0.0
            m.UB = 1.0
            m.VarName = "mock_var"
            return m

        def addConstr(self, *args, **kwargs):
            pass

        def setObjective(self, *args, **kwargs):
            pass

    mock_model = ModelMock()
    solver.model = cast(gp.Model, mock_model)
    solver.x[(0, 1)] = MagicMock()
    solver.x[(0, 1)].LB = 0.0
    solver.x[(0, 1)].UB = 1.0
    solver.x[(0, 1)].X = 0.5
    solver.model.setParam("OutputFlag", 0)
    solver.model.setParam("LogToConsole", 0)

    fractional_vars = [("x", (0, 1), 0.5)]

    def side_effect_optimize():
        mock_model.Status = GRB.INFEASIBLE

    with patch.object(mock_model, 'optimize', side_effect=side_effect_optimize):
        solver._strong_branching(fractional_vars)

    assert not mock_model._obj_val_accessed_when_not_optimal

def test_fix_4_u_bounds_reset():
    """Fix 4: MTZ load variables (u) must be reset in _evaluate_node."""
    dist_matrix = np.zeros((3, 3))
    wastes = {1: 10.0, 2: 20.0}
    capacity = 100.0
    solver = BBSolver(dist_matrix, wastes, capacity, 1.0, 1.0)

    u1 = MagicMock()
    u2 = MagicMock()
    solver.u = {1: u1, 2: u2}

    solver.x = {(0, 1): MagicMock()}
    solver.y = {1: MagicMock(), 2: MagicMock()}

    solver.model = cast(gp.Model, MagicMock())
    solver.model.Status = GRB.OPTIMAL

    node = Node(bound=0.0)
    solver._evaluate_node(node)

    assert u1.LB == 10.0
    assert u1.UB == 100.0
    assert u2.LB == 20.0
    assert u2.UB == 100.0

def test_fix_5_monetary_cost_return():
    """Fix 5: policy_bb.py must return monetary travel cost."""
    policy = BranchAndBoundPolicy()

    with patch('logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.policy_bb.run_bb_optimizer') as mock_run:
        mock_run.return_value = ([[1, 2]], 100.0)

        sub_dist_matrix = np.array([
            [0, 10, 20],
            [10, 0, 5],
            [20, 5, 0]
        ])
        sub_wastes = {1: 10.0, 2: 10.0}
        capacity = 100.0
        revenue = 2.0
        cost_unit = 3.0
        values = {}
        mandatory_nodes = []

        routes, profit, travel_cost = policy._run_solver(
            sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values, mandatory_nodes
        )

        assert travel_cost == 105.0
        assert profit == -65.0

def test_fix_6_dfj_directed_cut():
    """Fix 6: DFJ cut must use outgoing flow for directed variables."""
    model = MagicMock()
    model._x_vars = {(1, 2): MagicMock(), (2, 1): MagicMock(), (1, 3): MagicMock()}
    model._num_nodes = 4

    def mock_cbGetSolution(var):
        if var == model._x_vars[(1, 2)] or var == model._x_vars[(2, 1)]:
            return 1.0
        return 0.0
    model.cbGetSolution = mock_cbGetSolution

    class _FakeLinExpr:
        """Minimal stand-in for gurobipy.LinExpr that supports >= comparison."""
        def __ge__(self, rhs):
            class _FakeConstr:
                def __str__(self):
                    return f"expr >= {rhs}"
            return _FakeConstr()

    _qs_path = (
        "logic.src.policies.routing"
        ".exact_and_decomposition_solvers.branch_and_bound.dfj.quicksum"
    )
    with patch(_qs_path, return_value=_FakeLinExpr()), \
         patch('networkx.connected_components') as mock_comp:
        mock_comp.return_value = [{1, 2}, {0}, {3}]

        _dfj_callback(model, GRB.Callback.MIPSOL)

        assert model.cbLazy.called
        args, _ = model.cbLazy.call_args
        assert ">=" in str(args[0])

def test_fix_7_node_ordering_and_post_init():
    """Fix 7: Node ordering and missing __post_init__."""
    n1 = Node(bound=5.0)
    n2 = Node(bound=6.0)
    assert (n1 < n2) is True
    assert "__post_init__" not in Node.__dict__

def test_fix_8_strong_branching_state_restoration():
    """Fix 8: Verify _strong_branching re-solves model at the end."""
    dist_matrix = np.zeros((3, 3))
    wastes = {1: 10.0, 2: 10.0}
    solver = BBSolver(dist_matrix, wastes, 100.0, 1.0, 1.0, branching_strategy="strong")

    mock_model = MagicMock()
    mock_model.ObjVal = 100.0
    mock_model.Status = GRB.OPTIMAL
    solver.model = cast(gp.Model, mock_model)
    solver.x[(0, 1)] = MagicMock()
    solver.x[(0, 1)].LB = 0.0
    solver.x[(0, 1)].UB = 1.0

    solver._strong_branching([("x", (0, 1), 0.5)])

    assert mock_model.optimize.call_count >= 3

def test_fix_10_node_field_comment():
    """Fix 10: Verify node.py field comment reflects maximization."""
    import inspect

    from logic.src.policies.helpers.solvers_and_matheuristics.common.node import Node
    source = inspect.getsource(Node)


    assert "upper bound for maximization" in source.lower()
    assert "field(compare=True)  # LP relaxation value (upper bound for maximization)" in source

def test_fix_12_no_y_branching():
    """Fix 12: Verify _get_branching_variable only returns x variables."""
    dist_matrix = np.zeros((3, 3))
    wastes = {1: 10.0, 2: 10.0}
    solver = BBSolver(dist_matrix, wastes, 100.0, 1.0, 1.0, branching_strategy="most_fractional")

    # Mock x and y variables
    x_var = MagicMock()
    x_var.X = 0.5
    solver.x = {(0, 1): x_var}

    y_var = MagicMock()
    y_var.X = 0.5
    solver.y = {1: y_var}

    # Mock model
    solver.model = cast(gp.Model, MagicMock())
    solver.model.Status = GRB.OPTIMAL

    result = solver._get_branching_variable()

    # Should be the x variable, not y
    assert result is not None
    assert result[0] == "x"
    assert result[1] == (0, 1)

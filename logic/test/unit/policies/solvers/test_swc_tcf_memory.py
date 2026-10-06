"""Native TCF resource limits and native-model lifetime regressions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from logic.src.pipeline.simulations.solver_status import current_solver_status, reset_solver_status
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow import (
    gurobi,
)


class Expr:
    """Small license-free stand-in for Gurobi expressions."""

    X = 0.0

    def __add__(self, other):
        return self

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__
    __mul__ = __add__
    __rmul__ = __add__
    __eq__ = __add__
    __le__ = __add__


@pytest.fixture
def model(monkeypatch):
    result = MagicMock()
    result.Params = SimpleNamespace(Threads=0, SoftMemLimit=float("inf"))
    result.Status = gurobi.GRB.MEM_LIMIT
    result.SolCount = 0
    result.addVars.side_effect = lambda keys, **kw: {k: Expr() for k in keys}
    result.addVar.return_value = Expr()
    monkeypatch.setattr(gurobi.gp, "Model", lambda *a, **kw: result)
    monkeypatch.setattr(gurobi, "quicksum", lambda terms: sum(terms, Expr()))
    reset_solver_status()
    return result


def solve(values=None, env=None):
    return gurobi._run_gurobi_optimizer(
        np.array([50.0]),
        [[0.0, 1.0], [1.0, 0.0]],
        env,
        {"Omega": 0.1, "psi": 1.0, "Q": 100.0, "R": 1.0, "C": 1.0, **(values or {})},
        [1],
        [],
        number_vehicles=0,
    )


def test_defaults_bound_threads_and_memory(model):
    solve()
    assert model.Params.Threads == 2
    assert model.Params.SoftMemLimit == 5.0
    model.dispose.assert_called_once()


def test_limits_forwarded_and_stricter_environment_preserved(model):
    model.Params.Threads = 1
    model.Params.SoftMemLimit = 0.75
    solve({"gurobi_threads": 2, "gurobi_soft_mem_limit_gb": 1.0}, env=object())
    assert model.Params.Threads == 1
    assert model.Params.SoftMemLimit == 0.75
    model.dispose.assert_called_once()


def test_explicit_limits(model):
    solve({"gurobi_threads": 1, "gurobi_soft_mem_limit_gb": 0.5})
    assert model.Params.Threads == 1
    assert model.Params.SoftMemLimit == 0.5


@pytest.mark.parametrize("stage", ["addVars", "optimize"])
def test_native_model_released_on_failure(model, stage):
    getattr(model, stage).side_effect = RuntimeError("injected failure")
    with pytest.raises(RuntimeError, match="injected failure"):
        solve()
    model.dispose.assert_called_once()


def test_memory_limit_without_incumbent_is_visible(model):
    # No incumbent: the profitable collect-everything start (#41) is executed and the
    # memory stop stays visible in the day's status.
    assert solve() == ([0, 1, 0], pytest.approx(47.9), 2.0)
    assert current_solver_status() == "gurobi:MEM_LIMIT -> fallback:clarke_wright"
    model.dispose.assert_called_once()


def test_memory_limit_without_incumbent_keeps_an_empty_day_when_no_start_pays(model):
    # Nothing forced and the only start loses money: the empty plan is better.
    assert solve({"R": 0.01}) == ([0, 0], 0.0, 0.0)
    assert current_solver_status() == "gurobi:MEM_LIMIT"
    model.dispose.assert_called_once()


def test_memory_limit_keeps_incumbent(model):
    model.SolCount = 1
    model.ObjVal = 48.0
    model.Params.MIPGap = 0.01

    def variables(keys, **kwargs):
        result = {k: Expr() for k in keys}
        if kwargs.get("name") == "x":
            for variable in result.values():
                variable.X = 1.0
        return result

    model.addVars.side_effect = variables
    assert solve() == ([0, 1, 0], 48.0, 2.0)
    assert current_solver_status() == "gurobi:MEM_LIMIT"
    model.dispose.assert_called_once()


@pytest.mark.parametrize(
    "values",
    [
        {"gurobi_threads": 0},
        {"gurobi_threads": 1.5},
        {"gurobi_soft_mem_limit_gb": 0},
        {"gurobi_soft_mem_limit_gb": float("inf")},
    ],
)
def test_invalid_limits_rejected_before_build(model, values):
    with pytest.raises(ValueError):
        solve(values)
    model.addVars.assert_not_called()


def test_omegaconf_policy_resources_reach_native_model(model, monkeypatch):
    """The typed policy must retain resource controls through the dispatcher."""
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf import (
        SWCTCFPolicy,
    )
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "swc_tcf": {
                "Omega": 0.1,
                "psi": 1.0,
                "framework": "gurobi",
                "engine": "gurobi",
                "gurobi_threads": 1,
                "gurobi_soft_mem_limit_gb": 0.5,
            }
        }
    )
    policy = SWCTCFPolicy(config)
    from logic.src.pipeline.simulations import repository
    from logic.src.policies.route_construction.base import base_routing_policy

    monkeypatch.setattr(base_routing_policy, "load_area_and_waste_type_params", lambda *a: (100, 1, 1, 1, 1))
    monkeypatch.setattr(repository, "load_temporal_params", lambda: (8, 40, 0.1))
    _, _, _, values = policy._load_area_params("test", "plastic", config)
    policy._run_solver(np.array([[0.0, 1.0], [1.0, 0.0]]), {1: 50.0}, 100.0, 1.0, 1.0, values, [])
    assert model.Params.Threads == 1
    assert model.Params.SoftMemLimit == 0.5
    model.dispose.assert_called_once()


def test_infeasible_retry_releases_model(model):
    model.optimize.side_effect = lambda: setattr(model, "Status", gurobi.GRB.INFEASIBLE)
    with pytest.raises(RuntimeError, match="infeasible"):
        solve({"psi": 0.1})
    assert model.optimize.call_count == 2
    model.dispose.assert_called_once()

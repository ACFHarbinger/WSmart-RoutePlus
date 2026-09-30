"""Fleet fallbacks must share a budget and preserve unlimited behavior."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition import (
    ms_bpc_sp_engine as engine,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params import (
    MSBPCSPParams,
)


@pytest.mark.parametrize("fleet,expired", [(2, True), (None, False)])
def test_empty_result_does_not_restart_expired_or_unlimited_solve(monkeypatch, fleet, expired):
    monkeypatch.setattr(engine, "VRPPMasterProblem", MagicMock())
    ticks = iter([0.0] + [10.0 if expired else 0.0] * 10000)
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(ticks))
    select = Mock(return_value={1})
    monkeypatch.setattr(engine, "_select_nodes_knapsack", select)
    params = MSBPCSPParams(time_limit=1.0, max_bb_nodes=0)
    matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    routes, value = engine.run_ms_bpc_sp(matrix, {1: 1.0, 2: 1.0}, 10.0, 0.0, 1.0, params=params, vehicle_limit=fleet)
    assert routes == []
    assert select.call_count == 1, "retry must not refresh an expired budget or change unlimited-fleet execution"
    assert params.time_limit == 1.0


def test_retry_receives_only_remaining_budget(monkeypatch):
    original = engine.run_ms_bpc_sp
    ticks = iter([0.0] + [0.25] * 10000)
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(engine, "VRPPMasterProblem", MagicMock())
    monkeypatch.setattr(engine, "_select_nodes_knapsack", Mock(return_value={1}))
    retry = Mock(return_value=([], 0.0))
    monkeypatch.setattr(engine, "run_ms_bpc_sp", retry)
    params = MSBPCSPParams(time_limit=1.0, max_bb_nodes=0)
    matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    original(matrix, {1: 1.0, 2: 1.0}, 10.0, 0.0, 1.0, params=params, vehicle_limit=2)
    assert retry.call_count == 1
    assert retry.call_args.kwargs["params"].time_limit == pytest.approx(0.75)
    assert params.time_limit == 1.0

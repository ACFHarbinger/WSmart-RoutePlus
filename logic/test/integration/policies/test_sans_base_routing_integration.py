"""SANS adapter regressions and parity against the pre-migration execute path."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from logic.src.policies.route_construction.base import base_routing_policy
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search import dispatcher
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.revenue import (
    compute_waste_collection_revenue,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.solution_initialization import (
    _get_bin_stock,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params import (
    SANSParams,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans import (
    SANSPolicy,
)


class OriginalAdapter(SANSPolicy):
    """Execute body from dfb049e7e, using the SAME explicitly repaired dispatchers."""

    def execute(self, **kwargs):
        params = SANSParams.from_config(self._config or kwargs.get("config", {}).get("sans", {}))
        if params.engine == "og":
            return dispatcher.execute_og(self, params=params, **kwargs)
        return dispatcher.execute_new(self, params=params, **kwargs)


@pytest.fixture
def context(monkeypatch):
    # Base loader takes raw €/kg and resolves €/percentage-point once.
    monkeypatch.setattr(
        base_routing_policy, "load_area_and_waste_type_params", lambda *args: (500.0, 0.2, 10.0, 1.0, 100.0)
    )
    bins = SimpleNamespace(n=5, c=np.array([10.0, 20.0, 30.0, 40.0, 50.0]), means=np.zeros(5))
    return dict(
        bins=bins,
        coords=pd.DataFrame({"ID": range(6), "Lng": np.arange(6.0), "Lat": np.arange(6.0)}),
        distance_matrix=np.abs(np.subtract.outer(np.arange(6.0), np.arange(6.0))),
        mandatory=[3],
        seed=7,
        config={
            "sans": dict(
                vrpp=False,
                shift_hours=8.0,
                avg_speed_kmh=35.0,
                service_time_h=0.0,
                time_limit=10.0,
                T_init=1.0,
                T_min=0.5,
                alpha=0.1,
                iterations_per_T=2,
                combination=(2, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            )
        },
    )


@pytest.mark.parametrize("raw_revenue", [0.2, 0.4])
def test_legacy_units_overrides_data_and_all_routes(context, raw_revenue):
    context["config"]["sans"].update(engine="og", revenue=raw_revenue)
    data = pd.DataFrame({"#bin": range(6), "Stock": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0], "Accum_Rate": np.zeros(6)})
    context.update(new_data=data, policy_name="sans-audit", sample_id=9)
    expected_rate = raw_revenue * 10 * 100 / 100

    def solve(actual, coords, matrix, combination, mandatory, values, *args, **kwargs):
        assert actual is data
        assert mandatory == [3]
        assert values["R"] == expected_rate
        assert values["vehicle_capacity"] == 500
        assert _get_bin_stock(actual, 3, values["E"], values["B"]) == 30
        assert (
            compute_waste_collection_revenue([[0, 1, 0], [0, 3, 0]], actual, values["E"], values["B"], values["R"])
            == 40 * expected_rate
        )
        return [[0, 1, 0], [0, 3, 0]], 0.0, []

    with patch.object(dispatcher, "find_solutions", side_effect=solve) as solver:
        tour, cost, profit, _, _ = SANSPolicy().execute(**context)
    solver.assert_called_once()
    assert tour == [0, 1, 0, 3, 0]
    assert cost == 8
    assert profit == 40 * expected_rate - cost


@pytest.mark.parametrize("engine", ["new", "og"])
@pytest.mark.parametrize("vrpp", [False, True])
@pytest.mark.parametrize("seed", [7, 42])
def test_actual_engines_match_original_adapter(context, engine, vrpp, seed):
    context["config"]["sans"].update(engine=engine, vrpp=vrpp)
    context["seed"] = seed
    old = OriginalAdapter().execute(**context)
    new = SANSPolicy().execute(**context)
    assert old[:3] == new[:3]
    assert set(context["mandatory"]).issubset(new[0])
    assert len(new[0]) > 2
    assert new[4] is old[4]
    # Base execution intentionally creates/merges a construction metrics ledger.
    assert new[3] is not None


def test_full_bin_override_preserves_optional_profit(context):
    policy = SANSPolicy()
    matrix, wastes, indices, mandatory = policy._create_subset_problem(
        [3], context["distance_matrix"], context["bins"], use_all_bins=False
    )
    assert indices == list(range(6))
    assert mandatory == [3]
    assert matrix.shape == (6, 6)
    assert len(wastes) == 5
    with patch(
        "logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans.execute_new",
        return_value=([0, 2, 3, 0], 6.0, 94.0, None, None),
    ):
        result = policy.execute(**context)
    assert result[:3] == ([0, 2, 3, 0], 6.0, 94.0)


def test_legacy_failure_is_not_an_empty_success(context):
    context["config"]["sans"]["engine"] = "og"
    with (
        patch.object(dispatcher, "find_solutions", side_effect=RuntimeError("solver failed")),
        pytest.raises(RuntimeError, match="solver failed"),
    ):
        SANSPolicy().execute(**context)


def test_empty_mandatory(context):
    context["mandatory"] = []
    assert SANSPolicy().execute(**context)[:3] == ([0, 0], 0.0, 0.0)


def test_legacy_invalid_combination_reports_configuration_error(context):
    context["config"]["sans"].update(engine="og", combination="best")
    with pytest.raises(ValueError, match="seven-value"):
        SANSPolicy().execute(**context)


def test_collinear_overlap_is_not_a_crossing():
    from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
        _find_crossed_arcs,
    )

    cache = {}
    _find_crossed_arcs([0, 1, 2, 3, 4, 5, 0], [[i, i] for i in range(6)], 0, cache)
    assert cache[0] == []

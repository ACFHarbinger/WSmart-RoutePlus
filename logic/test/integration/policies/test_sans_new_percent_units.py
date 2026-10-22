"""SANS 'new' engine values collected waste in the area's percent units (it used V = 0)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from logic.src.configs.policies.routing.meta_heuristics.sans import SANSConfig
from logic.src.policies.routing.base import base_routing_policy
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search import dispatcher
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state import (
    compute_profit,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans import (
    SANSPolicy,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture
def area(monkeypatch):
    # Raw area values: capacity 500 %, revenue 0.2 EUR/kg, density 10, cost 1, bin volume 2.5.
    monkeypatch.setattr(
        base_routing_policy, "load_area_and_waste_type_params", lambda *args: (500.0, 0.2, 10.0, 1.0, 2.5)
    )


def test_config_no_longer_overrides_the_area_volume(area):
    _, _, _, values = SANSPolicy(SANSConfig())._load_area_params("Rio Maior", "plastic", {})
    assert not hasattr(SANSConfig(), "V")
    assert values["V"] == 2.5


def test_new_engine_scores_revenue_in_percent_units(area, monkeypatch):
    seen = {}

    def spy(routes, matrix, time_limit, id_to_index, data, Q, *args, **kwargs):
        # After Q: T_init, T_min, alpha, iterations_per_T, R, V, B, ...
        R, V, B = args[4], args[5], args[6]
        seen.update(Q=Q, R=R, V=V, B=B)
        stocks = dict(zip(data["#bin"], data["Stock"], strict=False))
        seen["profit"] = compute_profit([[0, 1, 2, 0]], matrix, id_to_index, data, Q, R, V, B, set(), stocks)
        return [[0, 1, 2, 0]], 0.0, 0.0, None, None

    monkeypatch.setattr(dispatcher, "improved_simulated_annealing", spy)
    bins = SimpleNamespace(n=3, c=np.array([40.0, 60.0, 10.0]), means=np.zeros(3))
    dispatcher.execute_new(
        SANSPolicy(),
        SANSConfig(time_limit=1.0),
        bins=bins,
        coords=pd.DataFrame({"ID": range(4), "Lng": np.arange(4.0), "Lat": np.arange(4.0)}),
        distance_matrix=np.zeros((4, 4)),
        mandatory=[1],
        seed=0,
    )
    assert seen["V"] * seen["B"] == pytest.approx(100.0)
    _, _, revenue, load = seen["profit"]
    assert load == pytest.approx(100.0)  # bins 1 and 2: 40 % + 60 % of one bin
    assert revenue == pytest.approx(100.0 * seen["R"]) and revenue > 0

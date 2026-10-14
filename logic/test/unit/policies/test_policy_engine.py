


from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.policy_bpc import (
    BPCPolicy,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf import (
    SWCTCFPolicy,
)
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.policy_alns import (
    ALNSPolicy,
)
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.policy_hgs import HGSPolicy
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans import (
    SANSPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]



class MockBins:
    def __init__(self, n=5):
        self.n = n
        self.c = np.ones(n) * 100.0  # Overflowing to trigger policy execution
        self.means = np.ones(n) * 1.0
        self.std = np.ones(n) * 0.1
        self.collectlevl = 90.0

@pytest.fixture
def mock_engine_data():
    n_bins = 5
    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    # Create fake new_data (DataFrame)
    new_data = pd.DataFrame({
        "ID": range(n_bins + 1),
        "#bin": range(n_bins + 1),
        "Stock": [0.0] * (n_bins + 1),
        "Accum_Rate": [0.0] * (n_bins + 1)
    })

    # Create fake coords
    coords = pd.DataFrame({
        "ID": range(n_bins + 1),
        "Lat": [0.0] * (n_bins + 1),
        "Lng": [0.0] * (n_bins + 1)
    })

    return {
        "bins": MockBins(n_bins),
        "distance_matrix": dist_matrix,
        "distancesC": dist_matrix.astype(np.int32),
        "waste_type": "plastic",
        "area": "riomaior",
        "mandatory": [1, 2, 3],  # Populate mandatory to prevent early returns
        "new_data": new_data,
        "coords": coords,
        "config": {
            "bpc": {"bcp_engine": "ortools"},
            "hgs": {"engine": None},
            "alns": {"engine": None},
            "swc_tcf": {},
            "n_encode_layers": 3,
            "n_encode_sublayers": 1,
            "n_decode_layers": 1,
            "n_heads": 8,
            "lookahead": {
                "sans": {},
                "Omega": 0.1
            }
        },
        "n_vehicles": 2,
        "model_env": MagicMock(),
        "graph_size": n_bins
    }

@pytest.mark.unit
def test_policy_factory_standardized():
    # Test that RoutingAlgorithmFactory correctly identifies policies from Registry
    p = RoutingAlgorithmFactory.get_adapter("hgs")
    assert isinstance(p, HGSPolicy)

@pytest.mark.unit
def test_bcp_engine_override(mock_engine_data):
    with patch("logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.policy_bpc.run_bpc") as mock_run:
        mock_run.return_value = ([[1, 0]], 10.0)

        policy = RoutingAlgorithmFactory.get_adapter("bpc")
        assert isinstance(policy, BPCPolicy)
        policy.execute(**mock_engine_data)

        # Verify run_bpc was called
        assert mock_run.called
        args, kwargs = mock_run.call_args
        # kwargs['env'] should be present
        assert "env" in kwargs

@pytest.mark.unit
def test_swc_tcf_engine_override(mocker, mock_engine_data):
    mocker.patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params",
                 return_value=(4000, 0.16, 21.0, 1.0, 2.5))

    with patch("logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.run_swc_tcf_optimizer") as mock_opt:
        mock_opt.return_value = ([0, 1, 0], 10.0, 5.0)

        policy = RoutingAlgorithmFactory.get_adapter("swc_tcf")
        assert isinstance(policy, SWCTCFPolicy)
        policy.execute(**mock_engine_data)

        assert mock_opt.called

@pytest.mark.unit
def test_hgs_engine_override(mock_engine_data):
    with patch("logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.policy_hgs.run_hgs") as mock_run:
        mock_run.return_value = ([[1, 0]], 10.0, 5.0)

        policy = RoutingAlgorithmFactory.get_adapter("hgs")
        assert isinstance(policy, HGSPolicy)
        policy.execute(**mock_engine_data)

        assert mock_run.called

@pytest.mark.unit
def test_alns_engine_override(mock_engine_data):
    with patch("logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.policy_alns.run_alns") as mock_run:
        mock_run.return_value = ([[1, 0]], 10.0, 5.0)

        # Set specific engine in config
        mock_engine_data["config"]["alns"]["engine"] = "ortools"

        policy = RoutingAlgorithmFactory.get_adapter("alns")
        assert isinstance(policy, ALNSPolicy)
        policy.execute(**mock_engine_data)

        assert mock_run.called
        # Verify it received the engine to run_alns
        args, _ = mock_run.call_args
        # run_alns(dist_matrix, wastes, max_capacity, R, C, values, ...)
        # values is the 6th arg (index 5)
        values_arg = args[5]
        assert values_arg["engine"] == "ortools"

def test_sans_execution(mock_engine_data):
    # Patch improved_simulated_annealing in its home module (or where it's used)
    with patch("logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.dispatcher.improved_simulated_annealing") as mock_sans, \
         patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params") as mock_load:

        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)
        mock_sans.return_value = ([[1, 0]], 10.0, 5.0, 2.0, 10.0)

        policy = RoutingAlgorithmFactory.get_adapter("sans")
        assert isinstance(policy, SANSPolicy)
        policy.execute(**mock_engine_data)

        assert mock_sans.called

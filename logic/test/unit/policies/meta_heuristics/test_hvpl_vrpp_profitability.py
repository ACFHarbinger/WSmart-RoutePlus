

import numpy as np
import pytest
from logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.policy_hvpl import (
    HVPLPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_subset_dimensions():
    # Use HVPLPolicy to test BaseRoutingPolicy methods
    policy = HVPLPolicy()

    # 10 bins total
    dist_matrix = np.random.rand(11, 11)
    bins_mock = type('obj', (object,), {'c': np.array([50.0] * 10)})
    mandatory = [1, 3, 5]

    # 1. Test use_all_bins=False (Restore Default behavior)
    # This should result in a 4x4 problem (depot + 3 selected nodes)
    sub_dm, sub_wastes, indices, mandatory = policy._create_subset_problem(
        mandatory, dist_matrix, bins_mock, use_all_bins=False
    )

    assert sub_dm.shape == (4, 4), f"Subset DM should be 4x4, got {sub_dm.shape}"
    assert len(sub_wastes) == 3
    assert mandatory == [1, 2, 3], f"Mandatory local indices should be [1, 2, 3], got {mandatory}"
    assert indices == [0, 1, 3, 5]

    # 2. Test use_all_bins=True (VRPP/Override behavior)
    # This should result in an 11x11 problem
    sub_dm_all, _, indices_all, mandatory_all = policy._create_subset_problem(
        [1, 3, 5], dist_matrix, bins_mock, use_all_bins=True
    )

    assert sub_dm_all.shape == (11, 11), f"VRPP DM should be 11x11, got {sub_dm_all.shape}"
    assert mandatory_all == [1, 3, 5], f"Mandatory global IDs should be [1, 3, 5], got {mandatory_all}"
    assert indices_all == list(range(11))

def test_hvpl_respects_subset():
    # Verify that HVPLPolicy.execute uses the subset logic
    # We will check if the default configuration results in a tour restricted to mandatory
    dist_matrix = np.ones((11, 11))
    np.fill_diagonal(dist_matrix, 0)
    {i: 50.0 for i in range(1, 11)}
    mandatory = [2, 4, 6]

    config = {
        "hvpl": {
            "max_iterations": 1,
            "n_teams": 1,
            "vrpp": False
        }
    }

    policy = HVPLPolicy(config=config)
    bins_mock = type('obj', (object,), {'c': np.array([50.0]*10)})

    tour, cost, profit, *_ = policy.execute(
        area="Rio Maior",
        waste_type="plastic",
        bins=bins_mock,
        distance_matrix=dist_matrix,
        mandatory=mandatory
    )

    visited = set(tour)
    visited.discard(0)
    # Even if ACO finds more profitable nodes, the SUBSET logic restricted it to mandatory!
    assert visited == {2, 4, 6}, f"HVPL should be restricted to mandatory subset, got {visited}"

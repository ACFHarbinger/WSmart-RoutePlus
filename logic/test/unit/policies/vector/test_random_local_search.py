

import pytest
import torch
from logic.src.policies.vector.random_local_search import (
    RandomLocalSearchPolicy,
)
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.mark.unit
def test_random_local_search_policy_basic():
    """Test the RandomLocalSearchPolicy forward pass with default settings."""
    device = "cpu"
    batch_size = 2
    num_nodes = 8

    td = TensorDict(
        {
            "locs": torch.rand(batch_size, num_nodes, 2, device=device),
            "waste": torch.rand(batch_size, num_nodes, device=device) * 0.2,
            "capacity": torch.ones(batch_size, device=device),
        },
        batch_size=[batch_size],
    )
    td["waste"][:, 0] = 0.0  # Depot

    policy = RandomLocalSearchPolicy(env_name="mvptp", n_iterations=10, device=device).to(device)

    # Mock environment
    class MockEnv:
        waste_weight = 1.0
        cost_weight = 1.0

    out = policy(td, env=MockEnv())

    assert "reward" in out
    assert "actions" in out
    assert out["reward"].shape == (batch_size,)
    assert out["actions"].dim() == 2
    assert out["actions"].shape[0] == batch_size
    # Actions should be routed (contain 0s)
    assert (out["actions"] == 0).any()


@pytest.mark.unit
def test_random_local_search_policy_custom_probs():
    """Test with custom operator probabilities (e.g. only 2-opt)."""
    device = "cpu"
    batch_size = 1
    num_nodes = 5

    td = TensorDict(
        {
            "locs": torch.rand(batch_size, num_nodes, 2, device=device),
            "waste": torch.zeros(batch_size, num_nodes, device=device),
            "capacity": torch.ones(batch_size, device=device),
        },
        batch_size=[batch_size],
    )

    # Only two_opt
    op_probs = {
        "two_opt": 1.0,
        "swap": 0.0,
        "relocate": 0.0,
        "two_opt_star": 0.0,
        "swap_star": 0.0,
        "three_opt": 0.0,
    }

    policy = RandomLocalSearchPolicy(env_name="mvptp", n_iterations=5, op_probs=op_probs, device=device).to(device)

    assert torch.allclose(
        policy.probs, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    )  # sorted order: relocate, swap, swap_star, three_opt, two_opt, two_opt_star

    # Mock environment
    class MockEnv:
        waste_weight = 1.0
        cost_weight = 1.0

    out = policy(td, env=MockEnv())
    assert out["actions"].shape[0] == batch_size

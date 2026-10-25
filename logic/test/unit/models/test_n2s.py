"""
Tests for N2S Model and Policy.
"""

import pytest
from logic.src.envs.tsp_kopt import TSPkoptEnv
from logic.src.models.core.n2s import N2S
from logic.src.models.core.n2s.policy import N2SPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_n2s_policy_forward():
    """Test N2SPolicy iterative loop with neighborhood mask."""
    env = TSPkoptEnv(num_loc=20)
    policy = N2SPolicy(embed_dim=128, k_neighbors=10)

    td = env.reset(batch_size=[2])

    # Run one step of policy
    out = policy(td, env, strategy="greedy", max_steps=3)

    assert out["actions"].shape == (2, 3, 4)  # batch, steps, 4 action components (i+, i-, j, k)

    # Check that it moves
    assert td["solution"].shape == (2, 21)


def test_n2s_model():
    """Test N2S model wrapper."""
    env = TSPkoptEnv(num_loc=10)
    model = N2S(env, embed_dim=64, k_neighbors=5)

    td = env.reset(batch_size=[1])

    # N2S wrapper should call policy.forward
    out = model(td, env, max_steps=2)

    assert "actions" in out.keys()
    assert "reward" in out.keys()
    assert out["reward"].shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])

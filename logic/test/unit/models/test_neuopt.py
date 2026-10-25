"""
Tests for NeuOpt Model and Policy.
"""

import pytest
from logic.src.envs.tsp_kopt import TSPkoptEnv
from logic.src.models.core.neuopt import NeuOpt
from logic.src.models.core.neuopt.policy import NeuOptPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_neuopt_policy_forward():
    """Test NeuOptPolicy iterative loop."""
    env = TSPkoptEnv(num_loc=20)
    policy = NeuOptPolicy(embed_dim=128, num_layers=2)

    td = env.reset(batch_size=[2])

    # Run one step of policy
    out = policy(td, env, strategy="greedy", max_steps=3)

    assert "actions" in out.keys()
    assert out["actions"].shape == (2, 3, 2)  # batch, steps, 2 nodes

    # Check that it moves
    assert td["solution"].shape == (2, 21)


def test_neuopt_model():
    """Test NeuOpt model wrapper."""
    env = TSPkoptEnv(num_loc=10)
    model = NeuOpt(env, embed_dim=64, num_layers=1)

    td = env.reset(batch_size=[1])

    # NeuOpt wrapper should call policy.forward
    out = model(td, env, max_steps=2)

    assert "actions" in out.keys()
    assert "reward" in out.keys()
    assert out["reward"].shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])

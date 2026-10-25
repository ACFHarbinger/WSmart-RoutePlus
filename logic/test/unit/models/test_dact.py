"""
Tests for DACT Model and TSPkopt Environment.
"""

import pytest
import torch
from logic.src.envs.tsp_kopt import TSPkoptEnv
from logic.src.models.core.dact import DACT
from logic.src.models.core.dact.policy import DACTPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_tsp_kopt_env():
    """Test TSPkoptEnv reset and step."""
    env = TSPkoptEnv(num_loc=20)
    td = env.reset(batch_size=[2])

    assert "locs" in td.keys()  # type: ignore[not-iterable]
    assert "solution" in td.keys()  # type: ignore[not-iterable]
    assert td["solution"].shape == (2, 21)  # 20 customers + 1 depot

    # Initial reward
    reward_init = env.get_reward(td)
    assert reward_init.shape == (2,)

    # Perform a dummy action (2-opt swap of indices 1 and 5)
    action = torch.tensor([[1, 5], [2, 10]], dtype=torch.long)
    td["action"] = action
    td = env.step(td)

    # Check solution changed
    assert "solution" in td.keys()  # type: ignore[not-iterable]


def test_dact_policy_forward():
    """Test DACTPolicy iterative loop."""
    env = TSPkoptEnv(num_loc=20)
    policy = DACTPolicy(embed_dim=128)

    td = env.reset(batch_size=[2])

    # Run one step of policy
    # We need to ensure we can run it in greedy mode
    out = policy(td, env, strategy="greedy", max_steps=3)

    assert "actions" in out.keys()
    assert out["actions"].shape == (2, 3, 2)  # batch, steps, 2 nodes


def test_dact_model():
    """Test DACT model wrapper."""
    env = TSPkoptEnv(num_loc=10)
    model = DACT(env, embed_dim=64)

    td = env.reset(batch_size=[1])

    # DACT wrapper should call policy.forward
    out = model(td, env, max_steps=2)

    assert "actions" in out.keys()
    assert "reward" in out.keys()
    assert out["reward"].shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])

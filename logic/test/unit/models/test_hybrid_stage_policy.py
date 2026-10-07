

import pytest
import torch
from logic.src.models.core.hybrid_attention_model.hybrid_two_step_policy import (
    HybridTwoStagePolicy,
    ImprovementStepDecoder,
)
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def dummy_td():
    batch_size = 2
    num_nodes = 10
    locs = torch.rand(batch_size, num_nodes, 2)
    return TensorDict(
        {
            "locs": locs,
            "waste": torch.rand(batch_size, num_nodes),
            "depot": torch.rand(batch_size, 2),
            "vehicle_capacity": torch.ones(batch_size),
            "done": torch.zeros(batch_size, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )

class MockEnv:
    def get_reward(self, td, actions):
        return torch.zeros(td.size(0))

    def step(self, td):
        # Mock step logic if needed, but HybridPolicy handles internal steps
        return td

@pytest.mark.unit
def test_improvement_step_decoder(dummy_td):
    embed_dim = 16
    n_operators = 22 # Updated for full suite
    decoder = ImprovementStepDecoder(embed_dim=embed_dim, n_operators=n_operators)

    batch_size = dummy_td.size(0)
    # Dummy embeddings: [B, N, Embed]
    embeddings = torch.randn(batch_size, 10, embed_dim)

    logits, mask = decoder(dummy_td, embeddings, None)

    assert logits.shape == (batch_size, n_operators)
    assert mask.shape == (batch_size, n_operators)

@pytest.mark.unit
def test_hybrid_two_stage_policy_forward(dummy_td):
    batch_size = dummy_td.size(0)
    embed_dim = 16

    policy = HybridTwoStagePolicy(
        env_name="mvptp",
        embed_dim=embed_dim,
        hidden_dim=16,
        n_encode_layers=2,
        n_heads=2,
        refine_steps=2
    )

    env = MockEnv()
    output = policy(dummy_td, env, strategy="greedy")

    assert "actions" in output
    assert "reward" in output
    assert "log_likelihood" in output
    assert output["actions"].shape == (batch_size, 10) # [B, N] tours
    assert output["reward"].shape == (batch_size,)

@pytest.mark.unit
def test_hybrid_two_stage_policy_sampling(dummy_td):
    embed_dim = 16
    policy = HybridTwoStagePolicy(
        env_name="mvptp",
        embed_dim=embed_dim,
        hidden_dim=16,
        n_encode_layers=2,
        n_heads=2,
        refine_steps=1
    )
    env = MockEnv()
    output = policy(dummy_td, env, strategy="sampling")
    assert "log_likelihood" in output
    assert output["log_likelihood"].requires_grad or output["log_likelihood"].shape == (dummy_td.size(0),)

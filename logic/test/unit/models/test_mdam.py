


from typing import Optional, cast
from unittest.mock import MagicMock

import pytest
import torch
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.subnets.decoders.mdam.decoder import MDAMDecoder
from logic.src.models.subnets.decoders.mdam.path import MDAMPath
from tensordict import TensorDict, TensorDictBase

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class MockEnv(RL4COEnvBase):
    name = "ptp"
    def __init__(self):
        self.device = torch.device("cpu")

    def _reset_instance(self, td: TensorDictBase) -> TensorDictBase:
        return td

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.zeros(td.batch_size, dtype=torch.float)

    def _get_action_mask(self, td: TensorDictBase) -> torch.Tensor:
        return torch.ones(td.batch_size, 5, dtype=torch.bool)

    def _step(self, td: TensorDictBase) -> TensorDictBase:
        # minimal step implementation
        next_td = td.clone()
        next_td["done"] = torch.ones(td.batch_size, dtype=torch.bool)
        return next_td

    def get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None):
        return self._get_reward(tensordict, actions)

@pytest.fixture
def mdam_decoder():
    return MDAMDecoder(
        embed_dim=16,
        num_heads=2,
        num_paths=2,
        env_name="ptp"
    )

def test_mdam_decoder_init(mdam_decoder):
    assert mdam_decoder.num_paths == 2
    assert len(mdam_decoder.paths) == 2
    assert isinstance(mdam_decoder.paths[0], MDAMPath)

def test_mdam_path_precompute():
    embed_dim = 16
    path = MDAMPath(embed_dim=embed_dim, env_name="ptp", num_heads=2)

    batch_size = 2
    num_nodes = 5
    h_embed = torch.randn(batch_size, num_nodes, embed_dim)

    cache = path.precompute(h_embed)

    assert cache.node_embeddings.shape == (batch_size, num_nodes, embed_dim)
    assert cache.graph_context.shape == (batch_size, 1, embed_dim)
    # Glimpse key/val should be (num_heads, batch, 1, num_nodes, key_dim)
    assert cast(torch.Tensor, cache.glimpse_key).shape == (2, batch_size, 1, num_nodes, embed_dim // 2)

def test_mdam_decoder_forward(mdam_decoder):
    batch_size = 2
    num_nodes = 5
    embed_dim = 16

    # Mock inputs
    td = TensorDict({
        "done": torch.zeros(batch_size, dtype=torch.bool),
        "action_mask": torch.ones(batch_size, num_nodes, dtype=torch.bool),
        "current_node": torch.zeros(batch_size, 1, dtype=torch.long),
        "locs": torch.randn(batch_size, num_nodes, 2), # GenericContextEmbedder might need this of it's used
        "depot": torch.randn(batch_size, 2), # GenericContextEmbedder might need this
    }, batch_size=[batch_size])

    embeddings = (
        torch.randn(batch_size, num_nodes, embed_dim), # h
        None,
        torch.randn(batch_size, num_nodes, embed_dim), # attn
        torch.randn(batch_size, num_nodes, embed_dim), # V
        torch.randn(batch_size, num_nodes, embed_dim) # h_old (not used heavily in basic test)
    )

    env = MagicMock()
    env.step.side_effect = lambda td: {"next": td.set("done", torch.ones(td.batch_size, dtype=torch.bool))}
    env.get_reward.return_value = torch.zeros(batch_size, dtype=torch.float)

    # Run forward
    reward, ll, kl, actions = mdam_decoder(td, embeddings, env, strategy="greedy")

    assert reward.shape == (batch_size, mdam_decoder.num_paths)
    assert ll.shape == (batch_size, mdam_decoder.num_paths)
    assert actions.dim() == 2 # (batch, seq_len)
    assert kl.dim() == 0 # scalar

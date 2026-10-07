"""
Tests for Transductive Models (Active Search, EAS).
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from logic.src.envs.routing.ptp import PTPEnv
from logic.src.models.common.transductive.active_search import ActiveSearch
from logic.src.models.common.transductive.eas import EAS
from logic.src.models.common.transductive.eas_embeddings import EASEmb
from logic.src.models.common.transductive.eas_layers import EASLay
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories import AttentionComponentFactory

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MagicMock(return_value=torch.randn(2, 10))
        self.decoder = MagicMock()
        # Mocking for EAS
        self.init_embedding = nn.Parameter(torch.randn(1, 10))
        self.init_proj = nn.Linear(10, 10)
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, td, env=None, **kwargs):
        # Return something TransductiveModel expects
        return {
            "reward": torch.randn(td.batch_size),
            "log_p": torch.randn(td.batch_size),
            "actions": torch.zeros(td.batch_size[0], 5, dtype=torch.long)
        }


def test_active_search_forward():
    """Test Active Search fine-tuning loop."""
    env = PTPEnv(num_loc=10)
    am = AttentionModel(embed_dim=64, hidden_dim=64, problem=env, component_factory=AttentionComponentFactory())

    # Active Search wrapper
    model = ActiveSearch(am, n_search_steps=2, optimizer_kwargs={"lr": 0.1})

    td = env.reset(batch_size=[1])

    # Initial state dict (to verify it restores)
    initial_state = {k: v.clone() for k, v in am.state_dict().items()}

    out = model(td, env)

    assert "reward" in out
    assert "actions" in out
    assert "search_history" in out
    assert len(out["search_history"]) == 2

    # Check that parameters are restored
    for k, v in am.state_dict().items():
        assert torch.equal(v, initial_state[k])


def test_eas_forward():
    """Test EAS selective fine-tuning."""
    env = PTPEnv(num_loc=10)
    am = AttentionModel(embed_dim=64, hidden_dim=64, problem=env, component_factory=AttentionComponentFactory())

    # EAS wrapper targeting only the projection layer
    model = EAS(am, n_search_steps=1, search_param_names=["init_proj"])

    td = env.reset(batch_size=[1])

    out = model(td, env)

    assert "reward" in out
    assert len(out["search_history"]) == 1


def test_eas_variants_params():
    """Test EASEmb and EASLay parameter selection."""
    model = MockPolicy()

    # EASEmb should target 'init_embedding'
    eas_emb = EASEmb(model)
    params = list(eas_emb._get_search_params())

    # In TransductiveModel, _get_search_params returns the filtered params
    assert len(params) == 1

    # EASLay should target 'init_proj' and 'layers.2'
    eas_lay = EASLay(model)
    params_lay = list(eas_lay._get_search_params())
    # init_proj has 2 params (weight, bias), layers.2 has 2 params
    assert len(params_lay) == 4


if __name__ == "__main__":
    pytest.main([__file__])

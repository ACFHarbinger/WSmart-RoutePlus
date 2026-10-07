"""Tests for CriticNetwork."""

from unittest.mock import MagicMock

import pytest
import torch
from logic.src.models.common.critic_network.model import LegacyCriticNetwork

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_critic_network_init():
    """Verify initialization of CriticNetwork."""
    problem = MagicMock()
    problem.NAME = "vrpp"
    component_factory = MagicMock()

    model = LegacyCriticNetwork(
        problem=problem, component_factory=component_factory, embed_dim=16, hidden_dim=16, n_layers=1, n_sublayers=1
    )
    assert model.embed_dim == 16
    assert model.is_vrpp


def test_critic_network_forward():
    """Verify forward flow logic."""
    problem = MagicMock()
    problem.NAME = "vrpp"
    component_factory = MagicMock()

    # Mock encoder
    mock_encoder = MagicMock(return_value=torch.randn(2, 5, 16))
    component_factory.create_encoder.return_value = mock_encoder

    model = LegacyCriticNetwork(
        problem=problem, component_factory=component_factory, embed_dim=16, hidden_dim=16, n_layers=1, n_sublayers=1
    )

    # Mock inputs
    inputs = {"depot": torch.randn(2, 2), "loc": torch.randn(2, 5, 2), "waste": torch.randn(2, 5)}

    # Mock context embedder result
    model._init_embed = MagicMock(return_value=torch.randn(2, 5, 3))

    out = model(inputs)
    assert out.shape == (2, 1)

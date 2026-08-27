"""Regression coverage for TTOP model-component registration."""

from unittest.mock import MagicMock

import pytest
from logic.src.models.common.critic_network.model import LegacyCriticNetwork
from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.edges import EDGE_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_ttop_uses_cvrpp_model_components():
    """TTOP is a temporal CVRPP and needs every shared model registry entry."""
    assert INIT_EMBEDDING_REGISTRY["ttop"] is INIT_EMBEDDING_REGISTRY["cvrpp"]
    assert CONTEXT_EMBEDDING_REGISTRY["ttop"] is CONTEXT_EMBEDDING_REGISTRY["cvrpp"]
    assert EDGE_EMBEDDING_REGISTRY["ttop"] is EDGE_EMBEDDING_REGISTRY["cvrpp"]
    assert STATE_EMBEDDING_REGISTRY["ttop"] is STATE_EMBEDDING_REGISTRY["cvrpp"]


def test_legacy_critic_accepts_ttop():
    """The deprecated compatibility critic remains usable by saved pipelines."""
    problem = MagicMock()
    problem.NAME = "ttop"
    model = LegacyCriticNetwork(
        problem=problem,
        component_factory=MagicMock(),
        embed_dim=16,
        hidden_dim=16,
        n_layers=1,
        n_sublayers=1,
    )
    assert model.is_vrpp

"""Regression coverage for CTOP model-component registration."""

from unittest.mock import MagicMock

import pytest
from logic.src.models.common.critic_network.model import LegacyCriticNetwork
from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.edges import EDGE_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_ctop_uses_cvrpp_model_components():
    """CTOP is a temporal CVRPP and needs every shared model registry entry."""
    assert INIT_EMBEDDING_REGISTRY["ctop"] is INIT_EMBEDDING_REGISTRY["cvrpp"]
    assert CONTEXT_EMBEDDING_REGISTRY["ctop"] is CONTEXT_EMBEDDING_REGISTRY["cvrpp"]
    assert EDGE_EMBEDDING_REGISTRY["ctop"] is EDGE_EMBEDDING_REGISTRY["cvrpp"]
    assert STATE_EMBEDDING_REGISTRY["ctop"] is STATE_EMBEDDING_REGISTRY["cvrpp"]


def test_legacy_critic_accepts_ctop():
    """The deprecated compatibility critic remains usable by saved pipelines."""
    problem = MagicMock()
    problem.NAME = "ctop"
    model = LegacyCriticNetwork(
        problem=problem,
        component_factory=MagicMock(),
        embed_dim=16,
        hidden_dim=16,
        n_layers=1,
        n_sublayers=1,
    )
    assert model.is_vrpp

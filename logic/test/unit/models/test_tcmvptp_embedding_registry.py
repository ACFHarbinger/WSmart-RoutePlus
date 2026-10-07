"""Regression coverage for TCMVPTP model-component registration."""

from unittest.mock import MagicMock

import pytest
from logic.src.models.common.critic_network.model import LegacyCriticNetwork
from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.edges import EDGE_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_ctop_uses_cvrpp_model_components():
    """TCMVPTP is a temporal MVPTP and needs every shared model registry entry."""
    assert INIT_EMBEDDING_REGISTRY["tcmvptp"] is INIT_EMBEDDING_REGISTRY["mvptp"]
    assert CONTEXT_EMBEDDING_REGISTRY["tcmvptp"] is CONTEXT_EMBEDDING_REGISTRY["mvptp"]
    assert EDGE_EMBEDDING_REGISTRY["tcmvptp"] is EDGE_EMBEDDING_REGISTRY["mvptp"]
    assert STATE_EMBEDDING_REGISTRY["tcmvptp"] is STATE_EMBEDDING_REGISTRY["mvptp"]


def test_legacy_critic_accepts_tcmvptp():
    """The deprecated compatibility critic remains usable by saved pipelines."""
    problem = MagicMock()
    problem.NAME = "tcmvptp"
    model = LegacyCriticNetwork(
        problem=problem,
        component_factory=MagicMock(),
        embed_dim=16,
        hidden_dim=16,
        n_layers=1,
        n_sublayers=1,
    )
    assert model.is_ptp

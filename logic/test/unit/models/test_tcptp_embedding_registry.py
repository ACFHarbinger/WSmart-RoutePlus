"""Regression coverage for TCPTP model-component registration."""

from unittest.mock import MagicMock

import pytest
from logic.src.models.common.critic_network.model import LegacyCriticNetwork
from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.edges import EDGE_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_ctop_uses_cvrpp_model_components():
    """TCPTP is a temporal CPTP and needs every shared model registry entry."""
    assert INIT_EMBEDDING_REGISTRY["tcptp"] is INIT_EMBEDDING_REGISTRY["cptp"]
    assert CONTEXT_EMBEDDING_REGISTRY["tcptp"] is CONTEXT_EMBEDDING_REGISTRY["cptp"]
    assert EDGE_EMBEDDING_REGISTRY["tcptp"] is EDGE_EMBEDDING_REGISTRY["cptp"]
    assert STATE_EMBEDDING_REGISTRY["tcptp"] is STATE_EMBEDDING_REGISTRY["cptp"]


def test_legacy_critic_accepts_tcptp():
    """The deprecated compatibility critic remains usable by saved pipelines."""
    problem = MagicMock()
    problem.NAME = "tcptp"
    model = LegacyCriticNetwork(
        problem=problem,
        component_factory=MagicMock(),
        embed_dim=16,
        hidden_dim=16,
        n_layers=1,
        n_sublayers=1,
    )
    assert model.is_ptp

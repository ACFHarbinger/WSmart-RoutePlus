"""Regression coverage for TCPTP model-component registration."""


import pytest
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

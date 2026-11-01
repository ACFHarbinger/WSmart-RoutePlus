"""Regression coverage for CMPTPTC model-component registration."""


import pytest
from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.edges import EDGE_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_ctop_uses_cvrpp_model_components():
    """CMPTPTC is a temporal CMPTP and needs every shared model registry entry."""
    assert INIT_EMBEDDING_REGISTRY["cmptptc"] is INIT_EMBEDDING_REGISTRY["cmptp"]
    assert CONTEXT_EMBEDDING_REGISTRY["cmptptc"] is CONTEXT_EMBEDDING_REGISTRY["cmptp"]
    assert EDGE_EMBEDDING_REGISTRY["cmptptc"] is EDGE_EMBEDDING_REGISTRY["cmptp"]
    assert STATE_EMBEDDING_REGISTRY["cmptptc"] is STATE_EMBEDDING_REGISTRY["cmptp"]

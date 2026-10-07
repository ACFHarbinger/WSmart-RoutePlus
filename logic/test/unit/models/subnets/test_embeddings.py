"""Tests for embedding modules."""

import pytest
import torch
from logic.src.models.subnets.embeddings.context.ptp import PTPContextEmbedder as CVRPPContext
from logic.src.models.subnets.embeddings.context.ptp import PTPContextEmbedder as VRPPContext
from logic.src.models.subnets.embeddings.dynamic import DynamicEmbedding
from logic.src.models.subnets.embeddings.static import StaticEmbedding
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestContextEmbedding:
    """Tests for ContextEmbeddings."""

    def test_vrpp_context(self):
        """Verify VRPPContext output shape."""
        batch = 2
        nodes = 5
        embed_dim = 16
        model = VRPPContext(embed_dim)

        embeddings = torch.randn(batch, nodes, embed_dim)
        td = TensorDict({
            "current_node": torch.zeros(batch, dtype=torch.long),
            "current_load": torch.rand(batch, 1)
        }, batch_size=batch)

        out = model(embeddings, td)
        assert out.shape == (batch, 1, embed_dim)

    def test_cvrp_context(self):
        """Verify CVRPContext (alias) works."""
        batch = 2
        nodes = 5
        embed_dim = 16
        model = CVRPPContext(embed_dim)

        embeddings = torch.randn(batch, nodes, embed_dim)
        td = TensorDict({
            "current_node": torch.zeros(batch, dtype=torch.long),
            "remaining_capacity": torch.rand(batch, 1)
        }, batch_size=batch)

        out = model(embeddings, td)
        assert out.shape == (batch, 1, embed_dim)

class TestDynamicEmbedding:
    """Tests for DynamicEmbeddings."""

    def test_static_embedding(self):
        """Verify StaticEmbedding returns 0."""
        model = StaticEmbedding(16)
        td = TensorDict({}, batch_size=2)
        out = model(td)
        assert len(out) == 3
        # Should be tensors of 0
        assert torch.equal(out[0], torch.tensor(0.))

    def test_dynamic_embedding_zeros(self):
        """Verify DynamicEmbedding returns 0 if no dynamic feature."""
        model = DynamicEmbedding(16)
        td = TensorDict({}, batch_size=2)
        out = model(td)
        assert torch.equal(out[0], torch.tensor(0.))

    def test_dynamic_embedding_update(self):
        """Verify DynamicEmbedding returns tensors if 'visited' is present."""
        batch = 2
        nodes = 5
        embed_dim = 16
        model = DynamicEmbedding(embed_dim, dynamic_node_dim=1)

        td = TensorDict({
            "visited": torch.randint(0, 2, (batch, nodes)).bool()
        }, batch_size=batch)

        gl_k, gl_v, l_k = model(td)

        assert torch.is_tensor(gl_k)
        assert gl_k.shape == (batch, nodes, embed_dim)
        # Check non-zero
        assert gl_k.abs().sum() > 0

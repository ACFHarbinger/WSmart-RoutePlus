"""Tests for neural network modules."""

from unittest.mock import MagicMock

import pytest
import torch
from logic.src.models.subnets.embeddings.positional import pos_init_embedding
from logic.src.models.subnets.embeddings.positional.absolute_positional_embedding import (
    AbsolutePositionalEmbedding,
)
from logic.src.models.subnets.embeddings.positional.cyclic_positional_embedding import (
    CyclicPositionalEmbedding,
)
from logic.src.models.subnets.modules.activation_function import ActivationFunction
from logic.src.models.subnets.modules.cross_attention import (
    MultiHeadCrossAttention,
)
from logic.src.models.subnets.modules.efficient_graph_convolution import (
    PYG_AVAILABLE,
    EfficientGraphConvolution,
)
from logic.src.models.subnets.modules.feed_forward import FeedForward
from logic.src.models.subnets.modules.gated_graph_convolution import GatedGraphConvolution
from logic.src.models.subnets.modules.graph_convolution import GraphConvolution
from logic.src.models.subnets.modules.multi_head_attention import MultiHeadAttention
from logic.src.models.subnets.modules.normalization import Normalization
from logic.src.models.subnets.modules.skip_connection import SkipConnection

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestActivationFunction:
    """Tests for activation function wrappers."""

    def test_relu(self):
        """Verifies ReLU activation."""
        model = ActivationFunction("relu")
        x = torch.tensor([-1.0, 1.0])
        out = model(x)
        assert torch.equal(out, torch.tensor([0.0, 1.0]))

    def test_threshold(self):
        """Verifies threshold activation."""
        model = ActivationFunction("relu", tval=0.5, rval=10.0)
        x = torch.tensor([0.0, 0.4, 0.6, 1.0])
        out = model(x)
        assert torch.equal(out, torch.tensor([0.0, 0.4, 10.0, 10.0]))



@pytest.mark.skipif(
    not PYG_AVAILABLE,
    reason="torch_geometric and torch_sparse are required for EGC",
)
class TestEfficientGraphConvolution:
    """Tests for EfficientGraphConvolution."""

    def test_init_and_forward(self):
        """Verifies initialization and forward flow."""
        feat = 16
        out_feat = 16
        model = EfficientGraphConvolution(feat, out_feat, n_heads=2, num_bases=2)

        batch = 2
        nodes = 4
        x = torch.randn(batch, nodes, feat)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        out = model(x, edge_index)
        assert out.shape == (batch, nodes, out_feat)


class TestFeedForward:
    """Tests for FeedForward (MLP) module."""

    def test_forward(self):
        """Verifies MLP forward flow."""
        model = FeedForward(10, 5)
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2, 5)


class TestGatedGraphConvolution:
    """Tests for GatedGraphConvolution."""

    def test_forward(self):
        """Verifies gated graph convolution logic."""
        hidden = 16
        model = GatedGraphConvolution(hidden, aggregation="mean")

        batch = 2
        nodes = 5
        h = torch.randn(batch, nodes, hidden)
        e = torch.randn(batch, nodes, nodes, hidden)
        mask = torch.ones(batch, nodes, nodes, dtype=torch.long)  # Fixed: requires long/bool

        h_out, e_out = model(h, e, mask)
        assert h_out.shape == (batch, nodes, hidden)
        assert e_out.shape == (batch, nodes, nodes, hidden)


class TestGraphConvolution:
    """Tests for GraphConvolution."""

    def test_forward_batched_mean(self, mock_node_features, mock_adj_matrix):
        """Verifies batched mean aggregation."""
        model = GraphConvolution(16, 16, aggregation="mean")
        h = mock_node_features
        mask = mock_adj_matrix

        out = model(h, mask)
        assert out.shape == (2, 5, 16)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_forward_2d_mask(self):
        """Verifies MHA with 2D mask."""
        dim = 16
        heads = 2
        model = MultiHeadAttention(heads, dim, embed_dim=dim)

        batch = 2
        nodes = 5
        q = torch.randn(batch, 1, dim)
        h = torch.randn(batch, nodes, dim)
        mask = torch.zeros(batch, nodes).bool()

        out = model(q, h, mask)
        assert out.shape == (batch, 1, dim)

    def test_forward_3d_mask(self):
        """Verifies MHA with 3D mask."""
        dim = 16
        heads = 2
        model = MultiHeadAttention(heads, dim, embed_dim=dim)
        batch = 2
        nodes = 5
        q = torch.randn(batch, 1, dim)
        h = torch.randn(batch, nodes, dim)
        mask = torch.zeros(batch, 1, nodes).bool()

        out = model(q, h, mask)
        assert out.shape == (batch, 1, dim)


class TestNormalization:
    """Tests for Normalization wrapper."""

    def test_group_norm(self):
        """Verifies GroupNorm support."""
        embed_dim = 16
        n_groups = 4
        model = Normalization(embed_dim, norm_name="group", n_groups=n_groups)

        # Batch, Seq, Dim (N, L, C structure handled by Normalization forward)
        x = torch.randn(2, 5, embed_dim)
        out = model(x)
        assert out.shape == x.shape

    def test_layer_norm(self):
        """Verifies LayerNorm support."""
        model = Normalization(16, norm_name="layer")
        x = torch.randn(2, 5, 16)
        out = model(x)
        assert out.shape == x.shape



class TestSkipConnection:
    """Tests for SkipConnection."""

    def test_forward_args(self):
        """Verifies argument propagation in skip connections."""
        inner = MagicMock()
        inner.return_value = torch.ones(1)
        model = SkipConnection(inner)
        input = torch.zeros(1)
        out = model(input, arg1="foo")

        inner.assert_called_with(input, arg1="foo")
        assert out.item() == 1.0


class TestPositionalEmbeddings:
    """Tests for Positional Embeddings."""

    def test_absolute_pe_shape(self):
        """Verify Absolute Positional Embedding output shape."""
        batch = 2
        seq_len = 10
        dim = 16
        model = AbsolutePositionalEmbedding(dim)
        x = torch.randn(batch, seq_len, dim)
        out = model(x)
        assert out.shape == (batch, seq_len, dim)
        # Check if PE is added (output should not be equal to input)
        assert not torch.equal(out, x)

    def test_cyclic_pe_shape(self):
        """Verify Cyclic Positional Embedding output shape."""
        batch = 2
        seq_len = 10
        dim = 16
        model = CyclicPositionalEmbedding(dim)
        x = torch.randn(batch, seq_len, dim)
        positions = torch.rand(batch, seq_len)  # normalized positions [0, 1]
        out = model(x, positions)
        assert out.shape == (batch, seq_len, dim)
        assert not torch.equal(out, x)

    def test_factory_dispatch(self):
        """Verify factory function returns correct classes."""
        ape = pos_init_embedding("APE", 16)
        assert isinstance(ape, AbsolutePositionalEmbedding)
        cpe = pos_init_embedding("CPE", 16)
        assert isinstance(cpe, CyclicPositionalEmbedding)


class TestMultiHeadCrossAttention:
    """Tests for MultiHeadCrossAttention."""

    def test_forward_shape(self):
        """Verify output shape."""
        batch = 2
        embed = 16
        heads = 4
        q_len = 5
        kv_len = 10
        model = MultiHeadCrossAttention(embed, heads)
        q = torch.randn(batch, q_len, embed)
        kv = torch.randn(batch, kv_len, embed)
        out = model(q, kv)
        assert out.shape == (batch, q_len, embed)

    def test_forward_with_mask(self):
        """Verify forward flow with mask."""
        batch = 2
        embed = 16
        heads = 4
        q_len = 5
        kv_len = 10
        model = MultiHeadCrossAttention(embed, heads)
        q = torch.randn(batch, q_len, embed)
        kv = torch.randn(batch, kv_len, embed)
        # Mask True = invalid
        mask = torch.zeros(batch, q_len, kv_len, dtype=torch.bool)
        mask[:, :, 0] = True  # Mask first element
        out = model(q, kv, mask)
        assert out.shape == (batch, q_len, embed)

    def test_manual_attention_match(self):
        """Verify manual attention matches SDPA (roughly)."""
        # Note: They won't be identical due to float precision and implementation details
        # but should be close.
        batch = 1
        embed = 16
        heads = 4
        q_len = 5
        kv_len = 10

        model_sdpa = MultiHeadCrossAttention(embed, heads)
        model_manual = MultiHeadCrossAttention(embed, heads, store_attn_weights=True)
        # Copy weights
        model_manual.load_state_dict(model_sdpa.state_dict())

        q = torch.randn(batch, q_len, embed)
        kv = torch.randn(batch, kv_len, embed)

        model_sdpa.eval()
        model_manual.eval()

        with torch.no_grad():
            out_sdpa = model_sdpa(q, kv)
            out_manual = model_manual(q, kv)

        assert torch.allclose(out_sdpa, out_manual, atol=1e-5)

"""Tests for neural sub-networks (encoders/decoders)."""

from typing import cast

import pytest
import torch
import torch.nn as nn
from logic.src.models.subnets.decoders.gat.decoder import GraphAttentionDecoder
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.models.subnets.decoders.ptr.decoder import PointerDecoder
from logic.src.models.subnets.encoders.gac.encoder import GraphAttConvEncoder
from logic.src.models.subnets.encoders.gat.encoder import GraphAttentionEncoder
from logic.src.models.subnets.encoders.gcn.encoder import GraphConvolutionEncoder
from logic.src.models.subnets.encoders.mlp.encoder import MLPEncoder
from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder
from logic.src.models.subnets.encoders.ptr.encoder import PointerEncoder
from logic.src.models.subnets.encoders.tgc.encoder import TransGraphConvEncoder
from logic.src.models.subnets.other.gru_fill_predictor import GatedRecurrentUnitFillPredictor

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestGACEncoder:
    """Tests for GraphAttConvEncoder."""

    def test_init(self):
        """Verifies initialization."""
        model = GraphAttConvEncoder(n_heads=2, embed_dim=16, n_layers=1, n_groups=4)
        assert isinstance(model, nn.Module)

    def test_forward(self):
        """Verifies forward flow."""
        embed_dim = 16
        model = GraphAttConvEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        edges = torch.randn(batch, graph_size, graph_size)

        output = model(x, edges)
        assert output.shape == (batch, graph_size, embed_dim)


class TestGATDecoder:
    """Tests for GraphAttentionDecoder."""

    def test_forward(self):
        """Verifies forward flow."""
        embed_dim = 16
        model = GraphAttentionDecoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_groups=4)
        batch = 2
        graph_size = 5

        # MHA expects q to have sequence dimension if it asserts dim(2)
        q = torch.randn(batch, embed_dim).unsqueeze(1)  # (batch, 1, embed_dim)
        h = torch.randn(batch, graph_size, embed_dim)
        mask = torch.zeros(batch, graph_size, dtype=torch.bool)

        output = model(q, h, mask)
        assert output.shape == (batch, 1, 2)


class TestGATEncoder:
    """Tests for GraphAttentionEncoder."""

    def test_forward(self):
        """Verifies forward flow."""
        embed_dim = 16
        model = GraphAttentionEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)

        output = model(x)
        assert output.shape == (batch, graph_size, embed_dim)

    def test_hyper_forward(self):
        """Verifies hyper-connection forward flow."""
        embed_dim = 16
        model = GraphAttentionEncoder(
            n_heads=2, embed_dim=embed_dim, n_layers=1, connection_type="static_hyper", expansion_rate=2
        )
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        output = model(x)
        assert output.shape == (batch, graph_size, embed_dim)


class TestMoEGraphAttentionEncoder:
    """Tests for MoEGraphAttentionEncoder."""

    def test_forward(self):
        """Verifies MoE encoder forward flow."""
        embed_dim = 16
        model = MoEGraphAttentionEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, num_experts=2, k=1)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        output = model(x)
        assert output.shape == (batch, graph_size, embed_dim)


class TestGCNEncoder:
    """Tests for GraphConvolutionEncoder."""

    def test_forward(self):
        """Verifies forward flow."""
        hidden = 16
        model = GraphConvolutionEncoder(n_layers=1, feed_forward_hidden=hidden, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, hidden)
        edges = torch.randint(0, 2, (batch, graph_size, graph_size))

        output = model(x, edges)
        assert output.shape == (batch, graph_size, hidden)


class TestGatedRecurrentUnitFillPredictor:
    """Tests for GatedRecurrentUnitFillPredictor."""

    def test_forward(self):
        """Verifies fill prediction."""
        hidden = 16
        model = GatedRecurrentUnitFillPredictor(input_dim=1, hidden_dim=hidden, num_layers=1, activation="relu")
        batch = 2
        seq = 5
        x = torch.randn(batch, seq, 1)

        output = model(x)
        assert output.shape == (batch, 1)


class TestMLPEncoder:
    """Tests for MLPEncoder."""

    def test_forward(self):
        """Verifies MLP encoder forward flow."""
        dim = 16
        model = MLPEncoder(n_layers=1, feed_forward_hidden=dim)
        batch = 2
        nodes = 5
        x = torch.randn(batch, nodes, dim)

        output = model(x)
        assert output.shape == (batch, nodes, dim)


class TestPointerDecoder:
    """Tests for PointerDecoder."""

    def test_forward(self):
        """Verifies pointer decoder logic."""
        embed_dim = 16
        hidden_dim = 16
        model = PointerDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            tanh_exploration=10,
            use_tanh=True,
        )

        batch = 2
        seq = 5
        decoder_input = torch.randn(batch, embed_dim)
        embedded_inputs = torch.randn(seq, batch, embed_dim)
        hidden = (torch.randn(batch, hidden_dim), torch.randn(batch, hidden_dim))
        context = torch.randn(seq, batch, hidden_dim)

        model.strategy = "greedy"

        (log_p, selections), hidden_out = model(decoder_input, embedded_inputs, hidden, context)

        assert log_p.shape == (batch, seq, seq)
        assert selections.shape == (batch, seq)


class TestPointerEncoder:
    """Tests for PointerEncoder."""

    def test_forward(self):
        """Verifies pointer encoder logic."""
        input_dim = 10
        hidden_dim = 16
        model = PointerEncoder(input_dim, hidden_dim)

        seq = 5
        batch = 2
        x = torch.randn(seq, batch, input_dim)
        hidden = model.init_hidden(hidden_dim)

        # Expand hidden
        h0 = hidden[0].unsqueeze(0).unsqueeze(1).expand(1, batch, hidden_dim).contiguous()
        c0 = hidden[1].unsqueeze(0).unsqueeze(1).expand(1, batch, hidden_dim).contiguous()

        out, _ = model(x, (h0, c0))
        assert out.shape == (seq, batch, hidden_dim)


class TestTGCEncoder:
    """Tests for TransGraphConvEncoder."""

    def test_forward(self):
        """Verifies transformer graph conv encoder."""
        embed_dim = 16
        model = TransGraphConvEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_sublayers=1, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        edges = torch.randint(0, 2, (batch, graph_size, graph_size), dtype=torch.bool)

        output = model(x, edges)
        assert output.shape == (batch, graph_size, embed_dim)


class TestGlimpseDecoder:
    """Tests for GlimpseDecoder."""

    def test_init(self):
        """Verifies initialization."""
        from unittest.mock import MagicMock

        problem = MagicMock()
        problem.NAME = "ptp"
        model = GlimpseDecoder(embed_dim=16, hidden_dim=16, problem=problem)
        assert isinstance(model, nn.Module)

    def test_precompute(self):
        """Verifies precomputation logic."""
        from unittest.mock import MagicMock

        problem = MagicMock()
        problem.NAME = "ptp"
        model = GlimpseDecoder(embed_dim=16, hidden_dim=16, problem=problem, n_heads=2)

        batch, nodes, dim = 2, 5, 16
        embeddings = torch.randn(batch, nodes, dim)
        fixed = model._precompute(embeddings)

        assert fixed.node_embeddings.shape == (batch, nodes, dim)
        # assert fixed.glimpse_key.shape == (2, batch, 1, nodes, 8) -> Implementation returns (batch, heads, nodes, key_dim)
        # make_heads returns [batch, n_heads, graph_size, key_size]
        assert cast(torch.Tensor, fixed.glimpse_key).shape == (batch, 2, nodes, 8)

    def test_get_parallel_step_context(self):
        """Verifies step context."""
        from unittest.mock import MagicMock

        problem = MagicMock()
        problem.NAME = "ptp"
        model = GlimpseDecoder(embed_dim=16, hidden_dim=16, problem=problem)

        batch, nodes, dim = 2, 5, 16
        embeddings = torch.randn(batch, nodes, dim)
        state = MagicMock()
        state.get_current_node.return_value = torch.ones(batch, 1, dtype=torch.long)

        ctx = model._get_parallel_step_context(embeddings, state)
        # Current implementation projects back to embed_dim and squeezes
        assert ctx.shape == (batch, dim)

    def test_select_node(self):
        """Verifies node selection logic."""
        from unittest.mock import MagicMock



        problem = MagicMock()
        problem.NAME = "ptp"
        model = GlimpseDecoder(embed_dim=16, hidden_dim=16, problem=problem)

        probs = torch.tensor([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        mask = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.bool)

        # Test greedy
        model.strategy = "greedy"
        selected = model._select_node(probs, mask)
        assert torch.equal(selected, torch.tensor([1, 2]))

        # Test sampling
        model.strategy = "sampling"
        selected = model._select_node(probs, mask)
        assert selected.shape == (2,)



import pytest
import torch
from logic.src.models.subnets.embeddings.context.ptp import PTPContextEmbedder

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestContextEmbedder:

    def test_vrpp_embedder_shapes(self):
        """Test PTP embedder output shapes."""
        embed_dim = 64
        model = PTPContextEmbedder(embed_dim=embed_dim, node_dim=3, temporal_horizon=0)

        input_data = {
            "loc": torch.rand(1, 5, 2),  # Test 'loc' vs 'locs' key fallback
            "depot": torch.rand(1, 2),
            "waste": torch.rand(1, 5),
        }

        embeddings = model.init_node_embeddings(input_data, temporal_features=False)

        # Batch=1, Graph=5 -> Total 6
        assert embeddings.shape == (1, 6, embed_dim)

        # Check step context dim
        # PTP: embed_dim + 2
        assert model.step_context_dim == embed_dim + 2



from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories import NeuralComponentFactory

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestAttentionModelStructure:
    @pytest.fixture
    def mock_problem(self):
        problem = MagicMock()
        problem.NAME = "ptp"
        return problem

    @pytest.fixture
    def mock_factory(self):
        factory = MagicMock(spec=NeuralComponentFactory)
        # Mock components - strict spec causes issues with custom methods like _calc_log_likelihood
        factory.create_encoder.return_value = MagicMock()
        factory.create_decoder.return_value = MagicMock()
        return factory

    def test_initialization_vrpp(self, mock_problem, mock_factory):
        """Test initialization for PTP context."""
        mock_problem.NAME = "ptp"

        model = AttentionModel(embed_dim=128, hidden_dim=64, problem=mock_problem, component_factory=mock_factory)

        assert model.is_ptp

    def test_forward_structure(self, mock_problem, mock_factory):
        """Test basic forward flow (mocked)."""
        model = AttentionModel(embed_dim=128, hidden_dim=64, problem=mock_problem, component_factory=mock_factory)

        # Mock embedder output
        model.context_embedder.forward = MagicMock(return_value=(torch.rand(1, 10, 128), None))

        # Mock decoder output
        # decoder(input, embeddings, ...) -> (log_p, pi, cost)
        log_p = torch.rand(1, 10)
        pi = torch.randint(0, 10, (1, 10))
        # Return expected cost 10.0
        cast(Any, model.decoder).return_value = (log_p, pi, torch.tensor([10.0]))
        # Mock calc_log_likelihood attached to decoder
        model.decoder._calc_log_likelihood.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # Mock problem cost calculation
        # get_costs(input, pi, ...) -> (cost, cost_dict, mask)
        mock_problem.get_costs.return_value = (torch.tensor([10.0]), {}, None)

        # Fake input
        dummy_input = {
            "locs": torch.rand(1, 10, 2),
            "depot": torch.rand(1, 2),
            "waste": torch.rand(1, 10),
            "dist": torch.rand(1, 10, 10),
        }

        # Run forward
        out = model(dummy_input)
        cost = out["cost"]

        assert cost.item() == 10.0
        # assert mock_problem.get_costs.called  # AttentionModel delegates cost calc to decoder
        assert model.context_embedder.forward.called
        assert model.decoder.called

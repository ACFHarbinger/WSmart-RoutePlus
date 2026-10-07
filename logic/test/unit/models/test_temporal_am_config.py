


from unittest.mock import MagicMock

import pytest
from logic.src.models.core.temporal_attention_model.model import TemporalAttentionModel
from logic.src.models.core.temporal_attention_model.policy import TemporalAMPolicy
from logic.src.models.subnets.other.gru_fill_predictor import GatedRecurrentUnitFillPredictor
from logic.src.models.subnets.other.lstm_fill_predictor import LongShortTermMemoryFillPredictor

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestTemporalAMConfig:
    @pytest.fixture
    def mock_factory(self):
        factory = MagicMock()
        return factory

    @pytest.fixture
    def mock_problem(self):
        return MagicMock()

    def test_model_init_gru(self, mock_factory, mock_problem):
        """Test model initialization with GRU predictor."""
        model = TemporalAttentionModel(
            embed_dim=128,
            hidden_dim=128,
            problem=mock_problem,
            component_factory=mock_factory,
            predictor_type="gru"
        )
        assert isinstance(model.fill_predictor, GatedRecurrentUnitFillPredictor)

    def test_model_init_lstm(self, mock_factory, mock_problem):
        """Test model initialization with LSTM predictor."""
        model = TemporalAttentionModel(
            embed_dim=128,
            hidden_dim=128,
            problem=mock_problem,
            component_factory=mock_factory,
            predictor_type="lstm"
        )
        assert isinstance(model.fill_predictor, LongShortTermMemoryFillPredictor)

    def test_policy_init_gru(self):
        """Test policy initialization with GRU predictor."""
        policy = TemporalAMPolicy(
            env_name="vrpp",
            embed_dim=128,
            hidden_dim=128,
            predictor_type="gru"
        )
        assert isinstance(policy.fill_predictor, GatedRecurrentUnitFillPredictor)

    def test_policy_init_lstm(self):
        """Test policy initialization with LSTM predictor."""
        policy = TemporalAMPolicy(
            env_name="vrpp",
            embed_dim=128,
            hidden_dim=128,
            predictor_type="lstm"
        )
        assert isinstance(policy.fill_predictor, LongShortTermMemoryFillPredictor)

    def test_policy_default(self):
        """Test policy initialization with default (GRU)."""
        policy = TemporalAMPolicy(
            env_name="vrpp",
            embed_dim=128,
            hidden_dim=128
        )
        assert isinstance(policy.fill_predictor, GatedRecurrentUnitFillPredictor)

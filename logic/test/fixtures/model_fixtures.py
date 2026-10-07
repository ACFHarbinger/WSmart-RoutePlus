"""
Fixtures for Neural Model testing (AM, TAM, GAT, etc.) and RL Training (PPO, GRPO).
"""

from unittest.mock import MagicMock

import pytest
import torch
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.core.temporal_attention_model.model import TemporalAttentionModel
from logic.src.models.meta.hrl_manager import MandatoryManager
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.models.subnets.factories.base import NeuralComponentFactory


@pytest.fixture
def mock_node_features():
    """Returns a random batch of node features for module testing."""
    batch_size = 2
    num_nodes = 5
    hidden_dim = 16
    return torch.randn(batch_size, num_nodes, hidden_dim)


@pytest.fixture
def mock_adj_matrix():
    """Returns a random adjacency matrix for module testing."""
    batch_size = 2
    num_nodes = 5
    return torch.ones(batch_size, num_nodes, num_nodes)


@pytest.fixture
def am_setup(mocker):
    """Fixture for AttentionModel"""
    mock_problem = mocker.MagicMock()
    mock_problem.NAME = "vrpp"
    mock_problem.get_costs.return_value = (torch.zeros(1), {}, None)

    mock_encoder = mocker.MagicMock()
    mock_encoder.embed_dim = 128
    mock_encoder.hidden_dim = 128

    # Needs to return tensor on call
    def mock_enc_fwd(x, edges=None, **kwargs):
        """Mock encoder forward pass."""
        batch, n, dim = x.size()
        return torch.zeros(batch, n, 128)  # hidden_dim

    mock_encoder.side_effect = mock_enc_fwd

    # Mock Factory
    class MockFactory(NeuralComponentFactory):
        """Mock factory for neural components."""

        def create_encoder(self, norm_config=None, activation_config=None, **kwargs):
            """Create mock encoder."""
            return mock_encoder

        def create_decoder(self, decoder_type="attention", norm_config=None, activation_config=None, **kwargs):
            """Create mock decoder."""
            m_dec = mocker.MagicMock(spec=GlimpseDecoder)
            m_dec.forward.side_effect = lambda input, embeddings, *args, **kwargs: (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                None,
            )
            return m_dec

    model = AttentionModel(
        embed_dim=128,
        hidden_dim=128,
        problem=mock_problem,
        component_factory=MockFactory(),
        n_encode_layers=1,
        n_heads=8,
        checkpoint_encoder=False,
    )
    return model


@pytest.fixture
def gat_lstm_setup():
    """Fixture for MandatoryManager"""
    manager = MandatoryManager(
        input_dim_static=2,
        input_dim_dynamic=10,
        hidden_dim=32,
        lstm_hidden=16,
        num_layers_gat=1,
        n_heads=4,
        dropout=0.1,
        device="cpu",
    )
    return manager


@pytest.fixture
def tam_setup(mocker):
    """Fixture for TemporalAttentionModel"""
    mock_problem = mocker.MagicMock()
    mock_problem.NAME = "vrpp"  # To trigger temporal features
    mock_problem.get_costs.return_value = (torch.zeros(1), {}, None)

    mock_encoder = mocker.MagicMock()
    mock_encoder.embed_dim = 128
    mock_encoder.hidden_dim = 128

    def mock_enc_fwd(x, edges=None, **kwargs):
        """Mock encoder forward pass."""
        batch, n, dim = x.size()
        return torch.zeros(batch, n, 128)

    mock_encoder.side_effect = mock_enc_fwd

    class MockActivationFunction(torch.nn.Module):
        """Mock activation function."""

        def __init__(self, *args, **kwargs):
            """Initialize mock activation function."""
            super().__init__()

        def forward(self, x):
            """Mock forward pass."""
            return x

    mocker.patch("logic.src.models.subnets.modules.ActivationFunction", new=MockActivationFunction)

    # Patch where used in TAM model module
    mocker.patch("logic.src.models.core.temporal_attention_model.model.LongShortTermMemoryFillPredictor", autospec=False)
    mock_grfp_cls = mocker.patch("logic.src.models.core.temporal_attention_model.model.GatedRecurrentUnitFillPredictor", autospec=False)

    mock_grfp = mock_grfp_cls.return_value

    def mock_grfp_fwd(x, h=None):
        """Mock GRFP forward pass."""
        return torch.zeros(x.shape[0], 1)

    mock_grfp.side_effect = mock_grfp_fwd

    # Mock Factory for TAM
    class MockTAMFactory(NeuralComponentFactory):
        """Mock factory for TAM components."""

        def create_encoder(self, norm_config=None, activation_config=None, **kwargs):
            """Create mock encoder."""
            return mock_encoder

        def create_decoder(self, decoder_type="attention", norm_config=None, activation_config=None, **kwargs):
            """Create mock decoder."""
            """Create mock decoder."""
            m_dec = mocker.MagicMock(spec=GlimpseDecoder)
            # Return log_p, pi, cost, td (4 values)
            m_dec.forward.side_effect = lambda input, embeddings, *args, **kwargs: (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                None,
            )
            return m_dec

    model = TemporalAttentionModel(
        embed_dim=128,
        hidden_dim=128,
        problem=mock_problem,
        component_factory=MockTAMFactory(),
        n_encode_layers=1,
        n_heads=8,
        temporal_horizon=5,
    )
    return model


@pytest.fixture
def mock_train_model(mocker, mock_torch_device):
    """
    Returns a mock PyTorch model for training tests.
    It mimics the signature: cost, log_likelihood, c_dict, pi = model(x, cost_weights, return_pi, pad)
    """
    mock_model = mocker.MagicMock(spec=torch.nn.Module)

    # Setup standard return values
    cost = torch.tensor([10.0, 12.0], device=mock_torch_device, requires_grad=True)
    log_likelihood = torch.tensor([-0.5, -0.6], device=mock_torch_device, requires_grad=True)
    c_dict = {
        "length": torch.tensor([5.0, 6.0], device=mock_torch_device),
        "waste": torch.tensor([50.0, 60.0], device=mock_torch_device),
        "overflows": torch.tensor([0.0, 1.0], device=mock_torch_device),
    }
    pi = torch.tensor([[0, 1, 0], [0, 2, 0]], device=mock_torch_device)

    # Configure return values on instance call
    mock_model.return_value = (cost, log_likelihood, c_dict, pi)

    # Allow .to(device) chaining
    mock_model.to.return_value = mock_model
    mock_model.module = mock_model
    mock_model.set_strategy = mocker.MagicMock()

    return mock_model


@pytest.fixture
def mock_optimizer(mocker):
    """Returns a mock optimizer."""
    mock_opt = mocker.MagicMock()
    mock_opt.param_groups = [{"params": [], "lr": 0.001}]
    return mock_opt


@pytest.fixture
def mock_baseline(mocker):
    """Returns a mock baseline."""
    mock_bl = mocker.MagicMock()
    mock_bl.wrap_dataset.side_effect = lambda dataset, policy=None, env=None: dataset  # Pass through
    mock_bl.unwrap_batch.side_effect = lambda x: (x, None)  # Simple unwrap
    # eval returns baseline_val (Tensor)
    mock_bl.eval.side_effect = lambda td, reward, env=None: torch.zeros_like(reward)
    return mock_bl


@pytest.fixture
def mock_ppo_deps(mocker):
    """Fixture specific for PPO training tests."""
    mock_model = MagicMock()
    mock_model.return_value = (
        torch.tensor([1.0, 1.0]),  # cost
        torch.tensor([-1.0, -1.0], requires_grad=True),  # ll
        {},  # cost_dict
        torch.tensor([[0, 1], [0, 1]]),  # pi
        torch.tensor([0.5, 0.5]),  # entropy
    )
    mock_model.to = lambda x: mock_model

    mock_optimizer = MagicMock()
    mock_baseline = MagicMock()
    mock_baseline.wrap_dataset = lambda dataset, policy=None, env=None: dataset
    mock_baseline.unwrap_batch = lambda x: (x, None)
    mock_baseline.eval.side_effect = lambda td, reward, env=None: torch.ones_like(reward)

    mock_problem = MagicMock()
    mock_problem.NAME = "cvrpp"
    mock_problem.get_costs.return_value = (torch.tensor([1.0, 1.0]), {}, None)

    dataset_list = [
        {
            "loc": torch.rand(10, 2),
            "waste": torch.rand(10),
            "hrl_mask": torch.zeros(10, 10),
            "full_mask": torch.zeros(10, 11),
        }
        for _ in range(4)
    ]
    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = lambda self, idx: dataset_list[idx]
    mock_dataset.__len__ = lambda self: len(dataset_list)
    mock_dataset.dist_matrix = None
    mock_dataset.has_dist = False

    return {
        "model": mock_model,
        "optimizer": mock_optimizer,
        "baseline": mock_baseline,
        "problem": mock_problem,
        "training_dataset": mock_dataset,
        "val_dataset": [],
    }


@pytest.fixture
def mock_dr_grpo_deps():
    """Fixture specific for DR-GRPO training tests."""
    model = MagicMock()

    def model_side_effect(input, return_pi=False, expert_pi=None, imitation_mode=False, **kwargs):
        """Mock side effect for DR-GRPO model call."""
        if isinstance(input, dict):
            first_val = next(iter(input.values()))
            current_batch_size = first_val.size(0)
        else:
            current_batch_size = 4
        seq_len = 5
        return (
            torch.randn(current_batch_size, requires_grad=True),
            torch.randn(current_batch_size, requires_grad=True),
            {"total": torch.randn(current_batch_size)},
            torch.randn(current_batch_size, seq_len),
            torch.randn(current_batch_size, requires_grad=True),
        )

    model.side_effect = model_side_effect
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()
    model.__call__ = MagicMock(side_effect=model_side_effect)
    model.strategy = "sampling"
    model.set_strategy = MagicMock()

    optimizer = MagicMock()
    baseline = MagicMock()
    baseline.wrap_dataset.side_effect = lambda dataset, policy=None, env=None: dataset
    baseline.unwrap_batch.side_effect = lambda x: (x, None)

    def baseline_eval_side_effect(td, reward, env=None):
        """Mock side effect for DR-GRPO baseline evaluation."""
        return torch.zeros_like(reward)

    baseline.eval.side_effect = baseline_eval_side_effect

    dataset = MagicMock()
    dataset.__len__.return_value = 4
    dataset.__getitem__ = MagicMock(return_value={"input": torch.tensor([1.0, 2.0])})

    problem = MagicMock()
    problem.NAME = "vrpp"

    return {
        "model": model,
        "optimizer": optimizer,
        "baseline": baseline,
        "training_dataset": dataset,
        "val_dataset": dataset,
        "problem": problem,
    }


@pytest.fixture
def mock_gspo_deps():
    """Fixture specific for GSPO training tests."""
    model = MagicMock()

    def model_side_effect(input, *args, **kwargs):
        """Mock side effect for GSPO model call."""
        if isinstance(input, dict):
            first_val = next(iter(input.values()))
            current_batch_size = first_val.size(0)
        else:
            current_batch_size = 4
        seq_len = 5
        return (
            torch.randn(current_batch_size, requires_grad=True),
            torch.randn(current_batch_size, requires_grad=True),
            {"total": torch.randn(current_batch_size)},
            torch.randn(current_batch_size, seq_len),
            torch.randn(current_batch_size, requires_grad=True),
        )

    model.side_effect = model_side_effect
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()

    optimizer = MagicMock()
    baseline = MagicMock()
    baseline.wrap_dataset.side_effect = lambda dataset, policy=None, env=None: dataset
    baseline.unwrap_batch.side_effect = lambda x: (x, None)
    baseline.eval.side_effect = lambda td, reward, env=None: torch.zeros_like(reward)

    dataset = MagicMock()
    dataset.__len__.return_value = 4
    dataset.__getitem__ = MagicMock(return_value={"input": torch.tensor([1])})

    problem = MagicMock()
    problem.NAME = "vrpp"

    return {
        "model": model,
        "optimizer": optimizer,
        "baseline": baseline,
        "training_dataset": dataset,
        "val_dataset": dataset,
        "problem": problem,
    }


@pytest.fixture
def mock_sapo_deps():
    """Fixture specific for SAPO training tests."""
    model = MagicMock()
    msg_len = 5

    def model_side_effect(input, *args, **kwargs):
        """Mock side effect for SAPO model call."""
        if isinstance(input, dict):
            first_val = next(iter(input.values()))
            current_batch_size = first_val.size(0)
        else:
            current_batch_size = 4
        return (
            torch.randn(current_batch_size, requires_grad=True),
            torch.randn(current_batch_size, requires_grad=True),
            {"total": torch.randn(current_batch_size)},
            torch.randn(current_batch_size, msg_len),
            torch.randn(current_batch_size, requires_grad=True),
        )

    model.side_effect = model_side_effect
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()

    optimizer = MagicMock()
    baseline = MagicMock()
    baseline.wrap_dataset.side_effect = lambda dataset, policy=None, env=None: dataset
    baseline.unwrap_batch.side_effect = lambda x: (x, None)
    baseline.eval.side_effect = lambda td, reward, env=None: torch.zeros_like(reward)

    dataset = MagicMock()
    dataset.__len__.return_value = 4
    dataset.__getitem__ = MagicMock(return_value={"input": torch.tensor([1])})

    problem = MagicMock()
    problem.NAME = "vrpp"

    return {
        "model": model,
        "optimizer": optimizer,
        "baseline": baseline,
        "training_dataset": dataset,
        "val_dataset": dataset,
        "problem": problem,
    }

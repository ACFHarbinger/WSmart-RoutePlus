"""Tests for Hypernetwork and HypernetworkOptimizer."""

import pytest
import torch
from logic.src.models.meta.hypernet.hypernetwork import HyperNetwork as Hypernetwork
from logic.src.models.meta.hypernet.optimizer import HyperNetworkOptimizer as HypernetworkOptimizer

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class MockProblem:
    """Mock problem class for testing."""
    def __init__(self, name="ptp"):
        self.NAME = name

@pytest.fixture
def ptp_problem():
    return MockProblem("ptp")

@pytest.fixture
def wc_problem():
    return MockProblem("mvptp")

class TestHypernetwork:
    """Tests for the Hypernetwork module."""

    def test_initialization_ptp(self, ptp_problem):
        """Test initialization with PTP problem dims."""
        model = Hypernetwork(input_dim=6, output_dim=6, n_days=31, embed_dim=8, hidden_dim=16)
        # input_dim = 6. embed_dim = 8. combined_dim = 6 + 8 = 14
        assert model.layers[0].in_features == 14
        assert model.layers[-1].out_features == 6

    def test_initialization_wc(self, wc_problem):
        """Test initialization with MVPTP problem dims."""
        model = Hypernetwork(input_dim=6, output_dim=3)
        assert model.output_dim == 3

    def test_forward_pass(self, ptp_problem):
        """Test forward pass output shapes and values."""
        batch_size = 4
        model = Hypernetwork(input_dim=6, output_dim=6, n_days=31, embed_dim=8, hidden_dim=16)

        # 3*2 = 6 metrics
        metrics = torch.rand(batch_size, 6)
        days = torch.randint(0, 31, (batch_size,))

        weights = model(metrics, days)

        assert weights.shape == (batch_size, 6)
        assert (weights >= 0).all()  # Softplus output

class TestHypernetworkOptimizer:
    """Tests for the HypernetworkOptimizer class."""

    @pytest.fixture
    def optimizer(self, ptp_problem):
        return HypernetworkOptimizer(
            cost_weight_keys=["km", "kg", "overflows"],
            constraint_value=1.0,
            device=torch.device("cpu"),
            problem=ptp_problem
        )

    def test_update_buffer(self, optimizer):
        """Test buffer management."""
        metrics = torch.rand(6)
        day = 5
        weights = torch.rand(3)
        performance = 10.5

        optimizer.update_buffer(metrics, day, weights, performance)
        assert len(optimizer.buffer) == 1
        assert optimizer.best_performance == 10.5
        assert torch.equal(optimizer.best_weights, weights)

    def test_train_step(self, optimizer):
        """Test training logic with enough samples."""
        # Fill buffer
        for i in range(12):
            optimizer.update_buffer(
                torch.rand(6),
                i,
                torch.rand(3),
                float(20 - i)
            )

        initial_params = [p.clone() for p in optimizer.hypernetwork.parameters()]
        optimizer.train(epochs=2)

        # Verify parameters updated
        any_changed = False
        for p1, p2 in zip(initial_params, optimizer.hypernetwork.parameters(), strict=False):
            if not torch.equal(p1, p2):
                any_changed = True
                break
        assert any_changed

    def test_get_weights_low_experience(self, optimizer):
        """Return default weights if buffer is too small."""
        default = {"km": 0.5, "kg": 0.5, "overflows": 0.0}
        weights = optimizer.get_weights({}, 10, default)
        assert weights == default

    def test_get_weights_generation(self, optimizer):
        """Test generating weights from hypernetwork."""
        # Fill buffer
        for i in range(6):
            optimizer.update_buffer(torch.rand(6), i, torch.rand(3), 1.0)

        all_costs = {
            "overflows": torch.tensor([1.0, 0.0]),
            "kg": torch.tensor([100.0, 50.0]),
            "km": torch.tensor([10.0, 5.0]),
            "kg_lost": torch.tensor([0.0, 0.0])
        }

        weights = optimizer.get_weights(all_costs, 10, None)
        assert isinstance(weights, dict)
        assert set(weights.keys()) == {"km", "kg", "overflows"}
        assert abs(sum(weights.values()) - 1.0) < 1e-5

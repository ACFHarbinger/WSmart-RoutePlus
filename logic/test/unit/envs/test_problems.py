"""Tests for problem environment physics (VRPP, CVRPP, etc)."""

from unittest.mock import patch

import pytest
import torch
from logic.src.envs import problems as problem_module
from logic.src.envs.problems import CVRPP, VRPP

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestVRPP:
    """Tests for the VRPP problem class."""

    @pytest.mark.unit
    def test_get_costs_basic(self):
        """Test cost calculation for a simple VRPP route."""
        # Inject globals
        problem_module.COST_KM = 1.0
        problem_module.REVENUE_KG = 0.1
        problem_module.BIN_CAPACITY = 100.0

        dataset = {
            "depot": torch.tensor([[0.0, 0.0]]),  # (1, 2) OK
            "loc": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),  # (1, 2, 2)
            "waste": torch.tensor([[10.0, 20.0]]),  # (1, 2) OK
            "max_waste": torch.tensor([100.0]),  # (1) OK
        }
        pi = torch.tensor([[1, 2]])  # (1, 2)

        neg_profit, c_dict, _ = VRPP.get_costs(dataset, pi, cw_dict=None)

        # Length calculation:
        # 0->1: sqrt(1^2) = 1.0
        # 1->2: sqrt(1^2) = 1.0
        # 2->0: sqrt(2^2) = 2.0
        # Total Length = 4.0
        assert torch.allclose(c_dict["length"], torch.tensor([4.0]))

        # Waste: 10 + 20 = 30.0
        assert torch.allclose(c_dict["waste"], torch.tensor([30.0]))

        # Profit (COST_KM=1, REVENUE=1.0)
        # negative_profit = Length * 1 - Waste * 1.0 = 4.0 - 30.0 = -26.0
        assert torch.allclose(neg_profit.flatten(), torch.tensor([-26.0]))

    @pytest.mark.unit
    def test_make_state_defaults(self):
        """Test proper initialization of defaults in make_state."""
        problem_module.COST_KM = 1.0
        problem_module.REVENUE_KG = 0.1
        problem_module.BIN_CAPACITY = 100.0

        with patch("logic.src.envs.problems.BaseProblem.make_state") as mock_init:
            VRPP.make_state(input_data="mock_input", profit_vars={"cost_km": 1.0})

            args, kwargs = mock_init.call_args
            assert "profit_vars" in kwargs
            assert kwargs["profit_vars"]["cost_km"] == 1.0

        # VRPPDataset is now handled via VRPPGenerator + TensorDictDataset
        from logic.src.envs.generators import VRPPGenerator

        generator = VRPPGenerator(num_loc=10)
        td = generator(batch_size=2)
        assert td.batch_size[0] == 2
        assert td["waste"].shape == (2, 10)


class TestStateVRPP:
    """Tests for StateVRPP logic."""

    @pytest.mark.unit
    def test_initialize(self):
        """Test initialization of StateVRPP."""
        # Input mock
        batch_size = 2
        n_loc = 3
        input_data = {
            "depot": torch.zeros(batch_size, 2),
            "loc": torch.rand(batch_size, n_loc, 2),
            "waste": torch.rand(batch_size, n_loc),
            "max_waste": torch.ones(batch_size) * 10.0,
        }

        state = VRPP.make_state(input_data)

        assert state.ids.shape == (batch_size, 1)
        # In new envs, visited is (B, N+1) and usually boolean
        assert state.td["visited"].shape == (batch_size, n_loc + 1)
        # waste should be padded with 0 (depot) -> n_loc + 1
        assert state.td["waste"].shape == (batch_size, n_loc + 1)
        assert state.td["waste"][:, 0].eq(0).all()
        assert state.td["collected_waste"].shape == (batch_size,)

    @pytest.mark.unit
    def test_update_and_mask(self):
        """Test state update and mask generation."""
        batch_size = 1
        input_data = {
            "depot": torch.zeros(batch_size, 2),
            "loc": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),  # (1, 2, 2)
            "waste": torch.tensor([[5.0, 6.0]]),
            "max_waste": torch.tensor([10.0]),
        }
        state = VRPP.make_state(input_data)

        # Test mask at start (depot is 0, others unvisited)
        # get_mask returns mask where True means INVALID
        mask = state.get_mask()
        assert mask[:, 0, 0].eq(0).all()  # Depot allowed

        # Action: Go to node 1 (index 1)
        selected = torch.tensor([1])  # (batch_size,)
        state = state.update(selected)

        # Check visited
        assert state.td["visited"][:, 1].item()

        # Check collected waste
        # Node 1 waste = 5.0. total = 5.0
        assert state.get_current_profit().item() == 5.0

        # Check lengths
        # 0->1 dist is 1.0.
        assert state.td["tour_length"].item() == 1.0

        # Check mask again
        mask = state.get_mask()
        assert mask[:, 0, 1].item()  # Node 1 visited, now infeasible
        assert not mask[:, 0, 2].item()  # Node 2 unvisited


class TestStateCVRPP:
    """Tests for StateCVRPP logic."""

    @pytest.mark.unit
    def test_capacity_mask(self):
        """Test that nodes exceeding remaining capacity are masked."""
        batch_size = 1
        n_loc = 2
        # Setup: Bin Capacity 10.
        # Node 1: Waste 6.
        # Node 2: Waste 6.
        # We can visit 1 OR 2, but not both (6+6 > 10).
        input_data = {
            "depot": torch.zeros(batch_size, 2),
            "loc": torch.rand(batch_size, n_loc, 2),
            "waste": torch.tensor([[6.0, 6.0]]),
            "max_waste": torch.ones(batch_size) * 10.0,  # Clamp limit
        }

        profit_vars = {
            "cost_km": 1.0,
            "revenue_kg": 1.0,
            "bin_capacity": 10.0,
            "vehicle_capacity": 10.0,
        }
        state = CVRPP.make_state(input_data, profit_vars=profit_vars)

        # Visit Node 1
        selected = torch.tensor([1])
        state = state.update(selected)

        # Node 2 (idx 2) should be masked because 6 (curr) + 6 (potential) = 12 > 10
        # mask is (B, 1, N+1) and True means INVALID
        assert state.get_mask()[:, 0, 2].item() == 1
        # Depot (idx 0) should be unmasked (allowed to return)
        assert state.get_mask()[:, 0, 0].item() == 0

    @pytest.mark.unit
    def test_depot_reset(self):
        """Test that visiting depot resets used_capacity."""
        batch_size = 1
        n_loc = 2
        input_data = {
            "depot": torch.zeros(batch_size, 2),
            "loc": torch.rand(batch_size, n_loc, 2),
            "waste": torch.tensor([[6.0, 6.0]]),
            "max_waste": torch.ones(batch_size, 1) * 10.0,
        }
        profit_vars = {
            "cost_km": 1.0,
            "revenue_kg": 1.0,
            "bin_capacity": 10.0,
            "vehicle_capacity": 10.0,
        }
        state = CVRPP.make_state(input_data, profit_vars=profit_vars)

        # 0 -> 1 (Load 6)
        state = state.update(torch.tensor([1]))
        assert state.td["collected"].item() == 6.0

        # 1 -> 0 (Depot) -> Should reset capacity
        state = state.update(torch.tensor([0]))
        assert state.td["collected"].item() == 0.0

        # 0 -> 2 (Load 6) -> Should be allowed now
        state = state.update(torch.tensor([2]))
        assert state.td["collected"].item() == 6.0
        assert state.td["collected_waste"].item() == 12.0


class TestCVRPP:
    """Tests for the CVRPP problem class."""

    @pytest.mark.unit
    def test_make_state_uses_cvrpp_state(self):
        """Test that CVRPP uses StateCVRPP."""
        problem_module.COST_KM = 1.0
        problem_module.REVENUE_KG = 0.1
        problem_module.BIN_CAPACITY = 100.0

        with patch("logic.src.envs.problems.BaseProblem.make_state") as mock_init:
            CVRPP.make_state(input_data="mock")
            mock_init.assert_called_once()

    @pytest.mark.unit
    def test_get_costs_multitrip(self):
        """Test CVRPP.get_costs with multi-trip capacity logic."""
        # Setup: Vehicle Capacity 100.
        problem_module.BIN_CAPACITY = 100.0
        problem_module.VEHICLE_CAPACITY = 100.0
        problem_module.COST_KM = 1.0
        problem_module.REVENUE_KG = 1.0

        batch_size = 1
        dataset = {
            "depot": torch.zeros(batch_size, 2),
            "loc": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),  # 1 and 2 away
            "waste": torch.tensor([[60.0, 60.0]]),  # Each 60. Total 120 > 100.
            "max_waste": torch.tensor([100.0]),
        }

        # Scenario 1: Visit 1 -> 2 directly (Total 120) -> Should Fail Assertion
        pi_fail = torch.tensor([[1, 2]])
        with pytest.raises(AssertionError, match="Used more than capacity"):
            CVRPP.get_costs(dataset, pi_fail, cw_dict=None)

        # Scenario 2: Visit 1 -> 0 (Depot) -> 2 (Total 60 reset 60)
        pi_route = torch.tensor([[1, 0, 2]])
        cost, c_dict, _ = CVRPP.get_costs(dataset, pi_route, cw_dict=None)

        # Check Profit
        # Length:
        # 0->1 (1.0)
        # 1->0 (1.0)
        # 0->2 (2.0)
        # 2->0 (2.0)
        # Total Length = 6.0
        # collected waste is waste at 1 and 2. Depot implies reset capacity not "collection of negative waste".
        # get_costs logic sums gathered waste. gather([1, 0, 2]) -> [60, 0, 60]. Sum = 120.
        # Profit = 120 - 6 = 114. Negative Profit = -114.
        assert torch.allclose(cost, torch.tensor([-114.0]))

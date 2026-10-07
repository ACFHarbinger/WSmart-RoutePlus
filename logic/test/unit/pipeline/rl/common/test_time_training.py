

from unittest.mock import MagicMock

import pytest
import torch
from logic.src.pipeline.rl.common.epoch import apply_time_step, prepare_epoch
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestTimeTraining:
    def test_apply_time_step_pre_generated(self):
        # Mock dataset and TensorDict with pre-generated 3D waste
        batch_size = 2
        num_nodes = 3  # + 1 depot = 4 cols
        num_days = 3

        # 3D Waste: [bs, num_days, num_nodes+1]
        waste_3d = torch.zeros(batch_size, num_days, num_nodes + 1)
        # Day 0: 0, 0.5, 0.5, 0.5
        waste_3d[:, 0, 1:] = 0.5
        # Day 1: 0, 0.3, 0.3, 0.3 (next day waste to add)
        waste_3d[:, 1, 1:] = 0.3
        # Day 2: 0, 0.2, 0.2, 0.2
        waste_3d[:, 2, 1:] = 0.2

        locs = torch.zeros(batch_size, num_nodes + 1, 2)

        td = TensorDict(
            {
                "waste": waste_3d,
                "locs": locs,
                "depot": torch.zeros(batch_size, 2),
                "capacity": torch.tensor([1.0, 1.0]),
            },
            batch_size=[batch_size],
        )

        dataset = MagicMock()
        dataset.data = td

        # Routes / Actions (visited nodes)
        # B0: visited 1, 2
        # B1: visited 3
        actions = [torch.tensor([[1, 2, 0], [3, 0, 0]])]

        # Apply time step from day 0 to day 1
        apply_time_step(dataset, actions, day=0, env=MagicMock())

        new_waste = td["waste"]
        current_day = td["current_day"]

        # Check day updated
        assert current_day[0].item() == 1
        assert current_day[1].item() == 1

        # Check logic:
        # B0 Node 1: Was 0.5 -> Reset to 0 -> Add Day 1 Waste (0.3) -> 0.3
        assert torch.isclose(new_waste[0, 1], torch.tensor(0.3))
        # B0 Node 2: Was 0.5 -> Reset to 0 -> Add Day 1 Waste (0.3) -> 0.3
        assert torch.isclose(new_waste[0, 2], torch.tensor(0.3))
        # B0 Node 3: Was 0.5 -> Not visited -> Add Day 1 Waste (0.3) -> 0.8
        assert torch.isclose(new_waste[0, 3], torch.tensor(0.8))

        # B1 Node 1: Was 0.5 -> Not visited -> Add Day 1 Waste (0.3) -> 0.8
        assert torch.isclose(new_waste[1, 1], torch.tensor(0.8))
        # B1 Node 2: Was 0.5 -> Not visited -> Add Day 1 Waste (0.3) -> 0.8
        assert torch.isclose(new_waste[1, 2], torch.tensor(0.8))
        # B1 Node 3: Was 0.5 -> Reset 0 -> Add Day 1 Waste (0.3) -> 0.3
        assert torch.isclose(new_waste[1, 3], torch.tensor(0.3))

    def test_apply_time_step_on_the_fly(self):
        # Mock dataset and TensorDict for on-the-fly generation (2D waste)
        batch_size = 2
        num_nodes = 3  # + 1 depot = 4 cols

        # Current waste:
        # B0: [0, 0.5, 0.5, 0.5]
        # B1: [0, 0.4, 0.4, 0.4]
        waste = torch.tensor([[0.0, 0.5, 0.5, 0.5], [0.0, 0.4, 0.4, 0.4]])
        locs = torch.zeros(batch_size, num_nodes + 1, 2)

        td = TensorDict(
            {
                "waste": waste.clone(),
                "locs": locs,
                "depot": torch.zeros(batch_size, 2),
            },
            batch_size=[batch_size],
        )

        dataset = MagicMock()
        dataset.data = td

        # Mock env and generator
        class DummyGenerator:
            def __init__(self):
                self.noise_variance = 0.01
                self.noise_mean = 0.0
                self.capacity = 1.0
                self.max_fill = 1.0
                self.called = False

            def _generate_fill_levels(self, batch_size):
                self.called = True
                return torch.tensor([[0.0, 0.3, 0.3, 0.3], [0.0, 0.3, 0.3, 0.3]])

        env = MagicMock()
        gen = DummyGenerator()
        env.generator = gen
        env.name = "ptp"

        # Actions
        # B0 visited 1, 2
        # B1 visited 3
        actions = [torch.tensor([[1, 2, 0], [3, 0, 0]])]

        apply_time_step(dataset, actions, day=0, env=env)

        new_waste = td["waste"]

        # Ensure generator was called
        assert env.generator.called

        # B0 Node 1: Reset to 0 -> Add Day 1 Waste (0.3); no noise is applied
        assert torch.isclose(new_waste[0, 1], torch.tensor(0.3))
        # Node 3 wasn't visited, carryover was 0.5. New total 0.8
        assert torch.isclose(new_waste[0, 3], torch.tensor(0.8))

    def test_prepare_epoch_time_metadata(self):
        # Test that current_day is injected when train_time is True
        td = TensorDict({}, batch_size=[1])
        dataset = MagicMock()
        dataset.data = td

        model = MagicMock()
        model.train_time = True

        prepare_epoch(model, MagicMock(), MagicMock(), dataset, epoch=5, phase="train")
        assert td["current_day"].item() == 5

    def test_prepare_epoch_no_time_metadata(self):
        # Test that current_day is NOT injected when train_time is False
        td = TensorDict({}, batch_size=[1])
        dataset = MagicMock()
        dataset.data = td

        model = MagicMock()
        model.train_time = False

        prepare_epoch(model, MagicMock(), MagicMock(), dataset, epoch=5, phase="train")
        assert "current_day" not in td.keys()

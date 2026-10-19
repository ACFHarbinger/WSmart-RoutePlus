

import pytest
import torch
from logic.src.pipeline.learning.common.reward_scaler import RewardScaler
from logic.src.pipeline.learning.common.reward_scaler_batch import BatchRewardScaler

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_reward_scaler_welford():
    scaler = RewardScaler(scale="norm", running_momentum=0.0)

    # Initial state
    assert scaler.mean == 0.0
    assert scaler.variance == 1.0  # default for count < 2

    # Update with some values
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler.update(data)

    assert scaler.mean == 3.0
    assert torch.allclose(torch.tensor(scaler.variance), data.var(unbiased=False))

    # Test scaling
    scaled = scaler(data, update=False)
    assert torch.allclose(scaled.mean(), torch.tensor(0.0), atol=1e-6)
    # torch.std uses unbiased variance by default, use correction=0 for population std
    assert torch.allclose(scaled.std(correction=0), torch.tensor(1.0), atol=1e-6)


def test_reward_scaler_ema():
    scaler = RewardScaler(scale="norm", running_momentum=0.5)

    data1 = torch.tensor([2.0, 2.0])
    scaler.update(data1)
    assert scaler.mean == 2.0

    data2 = torch.tensor([4.0, 4.0])
    scaler.update(data2)
    # alpha * 4 + (1-alpha) * 2 = 0.5 * 4 + 0.5 * 2 = 3.0
    assert scaler.mean == 3.0


def test_reward_scaler_modes():
    scaler_none = RewardScaler(scale="none")
    scaler_scale = RewardScaler(scale="scale")

    data = torch.tensor([10.0, 20.0])
    # Set stats manually for predictable results
    scaler_scale._mean = 0.0
    scaler_scale._m2 = 100.0 * 2  # Variance = 100, Std = 10
    scaler_scale._count = 2

    assert torch.allclose(scaler_none(data, update=False), data)
    assert torch.allclose(scaler_scale(data, update=False), torch.tensor([1.0, 2.0]))


def test_batch_reward_scaler():
    scaler = BatchRewardScaler()
    data = torch.tensor([1.0, 2.0, 3.0])
    scaled = scaler(data)

    assert torch.allclose(scaled.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(scaled.std(correction=0), torch.tensor(1.0), atol=1e-6)


def test_state_dict():
    scaler = RewardScaler(scale="norm")
    scaler.update(torch.tensor([1.0, 2.0, 3.0]))

    sd = scaler.state_dict()
    assert sd["count"] == 3
    assert sd["mean"] == 2.0

    new_scaler = RewardScaler()
    new_scaler.load_state_dict(sd)
    assert new_scaler.mean == 2.0
    assert new_scaler._count == 3

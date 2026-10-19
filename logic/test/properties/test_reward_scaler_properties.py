

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from logic.src.pipeline.learning.common.reward_scaler import RewardScaler

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@st.composite
def rewards_tensor(draw):
    batch_size = draw(st.integers(min_value=2, max_value=20))
    data = draw(arrays(dtype=float, shape=(batch_size,), elements=st.floats(min_value=-1e3, max_value=1e3)))
    # Avoid constant arrays (std=0) which might cause division by small eps checks
    # Though scaler handles eps.
    t = torch.tensor(data, dtype=torch.float32)
    return t


@pytest.mark.property
@given(rewards=rewards_tensor())
def test_batch_scaler_normalization_properties(rewards):
    """
    Test that BatchRewardScaler produces zero mean and unit variance outputs.
    """
    scaler = RewardScaler(scale="norm", eps=1e-8)
    scaled = scaler(rewards)

    # Check shape
    assert scaled.shape == rewards.shape

    # Check dist stats if raw std > eps
    if rewards.std(correction=0) > 1e-3:
        assert torch.abs(scaled.mean()) < 1e-3
        assert torch.abs(scaled.std(correction=0) - 1.0) < 1e-3


@pytest.mark.property
@given(rewards=rewards_tensor(), shift=st.floats(min_value=-100, max_value=100))
def test_batch_scaler_shift_invariance(rewards, shift):
    """
    Test that BatchRewardScaler is invariant to additive shifts.
    f(x + c) = f(x)
    """
    scaler1 = RewardScaler(scale="norm", eps=1e-8)
    out1 = scaler1(rewards)

    scaler2 = RewardScaler(scale="norm", eps=1e-8)
    out2 = scaler2(rewards + shift)

    # Should be close
    if rewards.std(correction=0) > 1e-2:
        assert torch.allclose(out1, out2, atol=1e-3)


@pytest.mark.property
@given(rewards=rewards_tensor(), scale=st.floats(min_value=0.1, max_value=10.0))
def test_batch_scaler_scale_invariance(rewards, scale):
    """
    Test that BatchRewardScaler handles multiplicative scaling (sign preserved).
    f(c * x) = f(x) if c > 0
    f(c * x) = -f(x) if c < 0 (but st generates positive scale here)
    """
    scaler1 = RewardScaler(scale="norm", eps=1e-8)
    out1 = scaler1(rewards)

    scaler2 = RewardScaler(scale="norm", eps=1e-8)
    out2 = scaler2(rewards * scale)

    if rewards.std(correction=0) > 1e-2:
        assert torch.allclose(out1, out2, atol=1e-3)


@pytest.mark.property
@given(rewards=rewards_tensor())
def test_online_scaler_update_consistency(rewards):
    """
    Test that RewardScaler stats update consistently with Welford logic.
    """
    scaler = RewardScaler(scale="norm")
    scaler.update(rewards)

    # Check computed stats match direct computation for single batch
    raw_mean = rewards.mean().item()
    rewards.var(correction=0 if scaler._count < 2 else 1).item()
    # Welford usually computes population variance or sample variance?
    # Logic in code: _m2 / _count. This is population variance.
    # PyTorch var uses sample variance (correction=1) by default.

    # For a single batch update starting from 0:
    # _count = N
    # _mean = batch_mean
    # _m2 = sum((x - mean)**2)
    # variance = _m2 / N = population variance

    # Use absolute and relative tolerance
    assert abs(scaler.mean - raw_mean) < 2e-4

    raw_pop_var = rewards.var(correction=0).item()

    # Use relative tolerance for large variance values
    if raw_pop_var > 1e-4:
        assert abs(scaler.variance - raw_pop_var) / raw_pop_var < 5e-4
    else:
        assert abs(scaler.variance - raw_pop_var) < 1e-4


@pytest.mark.property
@given(rewards=rewards_tensor())
def test_online_scaler_state_dict(rewards):
    """
    Test saving and loading state dict.
    """
    scaler = RewardScaler(scale="norm")
    scaler.update(rewards)

    state = scaler.state_dict()

    scaler2 = RewardScaler(scale="norm")
    scaler2.load_state_dict(state)

    assert scaler2._count == scaler._count
    assert scaler2._mean == scaler._mean
    assert scaler2._m2 == scaler._m2

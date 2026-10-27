"""Gamma._sample_tensor without a generator draws fresh samples from the global torch RNG."""

import pytest
import torch
from logic.src.data.distributions import Gamma

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_successive_unseeded_draws_differ():
    dist = Gamma(option=0)
    a = dist._sample_tensor((2, 10))
    b = dist._sample_tensor((2, 10))
    assert not torch.equal(a, b)


def test_unseeded_draws_follow_the_global_seed():
    dist = Gamma(option=0)
    torch.manual_seed(123)
    a = dist._sample_tensor((2, 10))
    torch.manual_seed(123)
    b = dist._sample_tensor((2, 10))
    torch.testing.assert_close(a, b)


def test_explicit_generator_is_still_used():
    dist = Gamma(option=0)
    a = dist._sample_tensor((2, 10), generator=torch.Generator().manual_seed(7))
    b = dist._sample_tensor((2, 10), generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)

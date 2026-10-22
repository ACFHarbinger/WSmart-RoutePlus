"""regular(): a collection period below one day means every day (it divided by zero)."""

import numpy as np
import pytest
import torch

from logic.src.policies.packing import spec
from logic.src.policies.packing.vectorized import spec as vspec

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize("frequency", [0.25, 0.5, 0.99])
@pytest.mark.parametrize("day", [1, 2, 3])
def test_subdaily_period_collects_every_eligible_bin_every_day(frequency, day):
    fills = np.array([10.0, 50.0, 90.0])
    np.testing.assert_array_equal(spec.regular(fills, day, frequency, min_fill=40.0), [False, True, True])
    out = vspec.regular(torch.tensor([[0.0, 0.1, 0.5, 0.9]]), day, frequency, min_fill=0.4)
    assert out.tolist() == [[False, False, True, True]]


@pytest.mark.parametrize("frequency", [0, -1, 1, 2, 3])
def test_scalar_and_vector_schedules_agree_for_whole_periods(frequency):
    fills = np.array([20.0, 80.0])
    for day in range(1, 8):
        scalar = spec.regular(fills, day, frequency)
        vector = vspec.regular(torch.tensor([[0.0, 0.2, 0.8]]), day, frequency)[0, 1:].numpy()
        np.testing.assert_array_equal(scalar, vector)

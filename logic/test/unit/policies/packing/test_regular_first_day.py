"""Regular with a delayed calendar: no day before ``first_day`` is scheduled, then every period."""

import numpy as np
import pytest
import torch
from logic.src.interfaces.context import PackingContext
from logic.src.policies.packing import spec
from logic.src.policies.packing.regular import RegularSelection
from logic.src.policies.vectorized.packing import spec as vspec
from logic.src.policies.vectorized.packing.regular import RegularSelector

pytestmark = [pytest.mark.unit, pytest.mark.fast]

FILLS = np.array([10.0, 50.0, 90.0])


def _scheduled(frequency: float, first_day: int, days=range(1, 16)):
    return [d for d in days if spec.regular(FILLS, d, frequency, first_day=first_day).any()]


@pytest.mark.parametrize("period", [3, 4, 5])
def test_first_collection_is_on_day_k(period):
    assert _scheduled(period, period) == list(range(period, 16, period))


def test_default_first_day_keeps_the_original_calendar():
    assert _scheduled(3, 1) == [1, 4, 7, 10, 13]
    assert _scheduled(3, 1) == [d for d in range(1, 16) if spec.regular(FILLS, d, 3).any()]


def test_subdaily_period_starts_on_first_day():
    assert _scheduled(0.5, 4, range(1, 8)) == [4, 5, 6, 7]


@pytest.mark.parametrize("frequency", [0.5, 1, 3, 4])
@pytest.mark.parametrize("first_day", [1, 2, 4])
def test_scalar_and_vector_agree(frequency, first_day):
    for day in range(1, 13):
        scalar = spec.regular(FILLS, day, frequency, min_fill=40.0, first_day=first_day)
        tensor = torch.tensor([[0.0, 0.1, 0.5, 0.9]])
        vector = vspec.regular(tensor, day, frequency, min_fill=0.4, first_day=first_day)[0, 1:].numpy()
        np.testing.assert_array_equal(scalar, vector)
        selector = RegularSelector(frequency=int(max(frequency, 0)), min_fill=0.4, first_day=first_day)
        if frequency >= 1:
            np.testing.assert_array_equal(selector.select(tensor, current_day=day)[0, 1:].numpy(), scalar)


def test_selection_reads_first_day_from_the_context():
    def pack(day):
        ctx = PackingContext(
            bin_ids=np.arange(3, dtype="int32"),
            current_fill=FILLS,
            current_day=day,
            threshold=4,
            min_fill=15.0,
            first_day=4,
        )
        return RegularSelection().pack(ctx)[0]

    assert [pack(d) for d in (1, 2, 3)] == [[], [], []]
    assert pack(4) == [2, 3]
    assert pack(5) == []
    assert pack(8) == [2, 3]

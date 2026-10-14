"""Multi-day datasets can start from a spread of bin states."""

import pytest
import torch
from logic.src.pipeline.rl.common.epoch import accumulate_initial_fill
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _dataset(batch: int = 6, days: int = 5, bins: int = 3) -> TensorDict:
    return TensorDict({"waste": torch.full((batch, days, bins), 0.3)}, batch_size=[batch])


def test_instances_start_with_one_to_max_days_of_waste() -> None:
    td = accumulate_initial_fill(_dataset(), max_days=3)

    # Instance i holds (i mod 3) + 1 days of 0.3 per bin, capped at a full bin.
    assert td["waste"][:, 0].tolist() == pytest.approx([0.3, 0.6, 0.9, 0.3, 0.6, 0.9])
    assert td["_waste_full"].shape == (6, 5, 3)


def test_fill_is_capped_at_a_full_bin_and_days_at_the_dataset_length() -> None:
    td = accumulate_initial_fill(_dataset(days=5), max_days=50)

    assert td["waste"].max().item() == pytest.approx(1.0)
    assert td["waste"][4, 0].item() == pytest.approx(1.0)


@pytest.mark.parametrize("max_days", [0, 1])
def test_one_day_leaves_the_dataset_untouched(max_days) -> None:
    td = accumulate_initial_fill(_dataset(), max_days=max_days)

    assert td["waste"].dim() == 3
    assert "_waste_full" not in td.keys()

"""Multi-day datasets can start from a spread of bin states."""

from typing import cast

import pytest
import torch
from logic.src.pipeline.learning.common.epoch import accumulate_initial_fill
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

    waste = cast(torch.Tensor, td["waste"])
    assert waste.max().item() == pytest.approx(1.0)
    assert waste[4, 0].item() == pytest.approx(1.0)


@pytest.mark.parametrize("max_days", [0, 1])
def test_one_day_leaves_the_dataset_untouched(max_days) -> None:
    td = accumulate_initial_fill(_dataset(), max_days=max_days)

    assert td["waste"].dim() == 3
    assert "_waste_full" not in td.keys()


def _series(batch: int = 400, days: int = 12, bins: int = 4) -> torch.Tensor:
    return torch.full((batch, days, bins), 0.05)


def test_refresh_redraws_about_the_requested_share_and_leaves_the_rest() -> None:
    from logic.src.pipeline.learning.common.epoch import refresh_fill

    fill = torch.full((400, 4), 0.77)
    new_fill, chosen = refresh_fill(fill, _series(), fraction=0.5, max_days=10, day=3)

    assert 0.4 < chosen.float().mean().item() < 0.6
    assert torch.equal(new_fill[~chosen], fill[~chosen])
    # Redrawn bins hold 1 to 10 days of 0.05 each.
    days = new_fill[chosen][:, 0] / 0.05
    assert days.min().item() == pytest.approx(1.0) and days.max().item() == pytest.approx(10.0)
    assert torch.allclose(days, days.round(), atol=1e-4)


def test_refresh_is_reproducible_per_day_and_off_at_zero() -> None:
    from logic.src.pipeline.learning.common.epoch import refresh_fill

    fill = torch.zeros(400, 4)
    first = refresh_fill(fill, _series(), 0.5, 10, day=3)
    again = refresh_fill(fill, _series(), 0.5, 10, day=3)
    other_day = refresh_fill(fill, _series(), 0.5, 10, day=4)

    assert torch.equal(first[0], again[0])
    assert not torch.equal(first[1], other_day[1])
    assert not refresh_fill(fill, _series(), 0.0, 10, day=3)[1].any()

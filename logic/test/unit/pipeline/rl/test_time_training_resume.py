"""A resumed time-training run continues from the bin fill stored in its checkpoint."""

import types

import pytest
import torch
from logic.src.pipeline.rl.common.base.module import RL4COLitModule
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _Dataset:
    def __init__(self, td):
        self.data = td


def _module(fill: torch.Tensor):
    """Stands in for a training module: only the attributes the resume hooks read."""
    td = TensorDict({"waste": fill.clone(), "_waste_full": torch.rand(4, 31, 3)}, batch_size=[4])
    mod = types.SimpleNamespace(train_time=True, train_dataset=_Dataset(td), baseline=object())
    for name in ("_time_training_td", "on_save_checkpoint", "on_load_checkpoint", "_restore_time_training_fill"):
        setattr(mod, name, types.MethodType(getattr(RL4COLitModule, name), mod))
    return mod


def test_fill_round_trips_through_the_checkpoint() -> None:
    fill = torch.rand(4, 3)
    checkpoint: dict = {}
    _module(fill).on_save_checkpoint(checkpoint)

    resumed = _module(torch.zeros(4, 3))
    resumed.on_load_checkpoint(checkpoint)
    resumed._restore_time_training_fill()

    assert torch.equal(resumed.train_dataset.data["waste"], fill)
    # Restored once: a later epoch start leaves the evolving fill alone.
    resumed.train_dataset.data["waste"] = torch.ones(4, 3)
    resumed._restore_time_training_fill()
    assert torch.equal(resumed.train_dataset.data["waste"], torch.ones(4, 3))


def test_checkpoint_without_a_fill_leaves_the_dataset_unchanged() -> None:
    start = torch.rand(4, 3)
    resumed = _module(start)
    resumed.on_load_checkpoint({})
    resumed._restore_time_training_fill()
    assert torch.equal(resumed.train_dataset.data["waste"], start)


def test_fill_of_another_dataset_size_is_not_restored() -> None:
    checkpoint = {"time_training_fill": {"key": "waste", "fill": torch.rand(7, 3)}}
    start = torch.rand(4, 3)
    resumed = _module(start)
    resumed.on_load_checkpoint(checkpoint)
    resumed._restore_time_training_fill()
    assert torch.equal(resumed.train_dataset.data["waste"], start)

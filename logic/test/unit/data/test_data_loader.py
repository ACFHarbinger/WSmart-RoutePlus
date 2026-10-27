"""Tests for data_utils.py."""

from typing import cast

import pytest
import torch
from logic.src.utils.data.loader import check_extension, collate_fn, load_td_dataset, save_td_dataset
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_check_extension():
    """Verify extension checking."""
    assert check_extension("test", ".pkl") == "test.pkl"
    assert check_extension("test.pkl", ".pkl") == "test.pkl"
    assert check_extension("test.td", ".td") == "test.td"


def test_save_load_td_dataset(tmp_path):
    """Verify saving and loading TensorDict datasets."""
    td = TensorDict({"a": torch.randn(2)}, batch_size=[2])
    path = str(tmp_path / "data.td")
    save_td_dataset(td, path)

    loaded = load_td_dataset(path)
    assert torch.allclose(cast(torch.Tensor, td["a"]), cast(torch.Tensor, loaded["a"]))


def test_collate_fn():
    """Verify custom collation logic."""
    batch = [{"val": torch.tensor(1.0), "none": None}, {"val": torch.tensor(2.0), "none": None}]
    collated = collate_fn(batch)
    assert "val" in collated
    assert "none" not in collated
    assert torch.allclose(collated["val"], torch.tensor([1.0, 2.0]))

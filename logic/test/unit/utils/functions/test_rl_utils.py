"""Unit tests for rl.py."""

from typing import Any, cast

import pytest
import torch
from logic.src.utils.functions.rl import ensure_tensordict
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_ensure_tensordict_from_dict():
    """Test converting a simple dict to TensorDict."""
    batch = {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}
    td = ensure_tensordict(batch)
    assert isinstance(td, TensorDict)
    assert td.batch_size == torch.Size([2])
    assert torch.equal(cast(torch.Tensor, td["a"]), batch["a"])


def test_ensure_tensordict_already_td():
    """Test when input is already a TensorDict."""
    td_in = TensorDict({"a": torch.tensor([1])}, batch_size=[1])
    td_out = ensure_tensordict(td_in)
    assert td_out is td_in


def test_ensure_tensordict_wrapped_data():
    """Test handling 'data' wrapper from BaselineDataset."""
    batch = {"data": {"x": torch.tensor([10, 20])}}
    td = ensure_tensordict(batch)
    assert isinstance(td, TensorDict)
    assert td.batch_size == torch.Size([2])
    assert torch.equal(cast(torch.Tensor, td["x"]), torch.tensor([10, 20]))


def test_ensure_tensordict_wrapped_td():
    """Test when 'data' key already contains a TensorDict."""
    inner_td = TensorDict({"x": torch.tensor([1, 2])}, batch_size=[2])
    batch = {"data": inner_td}
    td = ensure_tensordict(batch)
    assert td is inner_td


def test_ensure_tensordict_fallback():
    """Test fallback for non-dict types."""
    val = [1, 2, 3]
    td = ensure_tensordict(cast(Any, val))  # runtime fallback wraps a non-dict batch
    assert td.batch_size == torch.Size([3])
    assert "data" in td.keys()
    assert td["data"].tolist() == val


def test_ensure_tensordict_zero_dim():
    """Test with zero-dimensional values."""
    batch = {"a": torch.tensor(1)}
    td = ensure_tensordict(batch)
    assert td.batch_size == torch.Size([])
    assert td["a"] == 1


def test_ensure_tensordict_device():
    """Test device movement."""
    # We use CPU as target if available or just check the .to call if mocked
    # But since we are in a real env, let's just use CPU to be safe and verify.
    batch = {"a": torch.tensor([1])}
    device = torch.device("cpu")
    td = ensure_tensordict(batch, device=device)
    assert td.device == device

"""Property-based tests for Boolean mask encoding and bit-packing utilities (§B.1 Option D)."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from logic.src.utils.functions.boolmask import (
    _mask_bool2byte,
    _mask_byte2bool,
    _pad_mask,
    mask_bool2long,
    mask_long2bool,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.property]


@st.composite
def boolean_mask_tensor(draw, batch_size=None, n_nodes=None):
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=8))
    if n_nodes is None:
        n_nodes = draw(st.integers(min_value=1, max_value=64))

    data = draw(arrays(dtype=bool, shape=(batch_size, n_nodes)))
    return torch.tensor(data, dtype=torch.uint8)


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(mask=boolean_mask_tensor())
def test_pad_mask_divisible_by_8(mask: torch.Tensor) -> None:
    """Invariant: _pad_mask always produces a last dimension divisible by 8 and preserves original prefix."""
    orig_n = mask.size(-1)
    padded, n_bytes = _pad_mask(mask)

    assert padded.size(-1) % 8 == 0
    assert padded.size(-1) == n_bytes * 8
    assert padded.size(-1) >= orig_n
    assert torch.equal(padded[..., :orig_n], mask)


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(mask=boolean_mask_tensor())
def test_mask_bool2byte_roundtrip(mask: torch.Tensor) -> None:
    """Invariant: bool -> byte -> bool roundtrip preserves exact boolean state."""
    orig_n = mask.size(-1)
    byte_mask = _mask_bool2byte(mask)
    recovered = _mask_byte2bool(byte_mask, n=orig_n)

    expected = mask > 0
    assert torch.equal(recovered, expected)


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(mask=boolean_mask_tensor())
def test_mask_bool2long_roundtrip(mask: torch.Tensor) -> None:
    """Invariant: bool -> long -> bool roundtrip preserves exact boolean state across any node size."""
    orig_n = mask.size(-1)
    long_mask = mask_bool2long(mask)
    recovered = mask_long2bool(long_mask, n=orig_n)

    expected = mask > 0
    assert torch.equal(recovered, expected)

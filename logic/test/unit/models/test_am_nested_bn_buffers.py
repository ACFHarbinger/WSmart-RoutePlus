"""AttentionModel keeps running BatchNorm stats when it is loaded nested under a prefix."""

import pytest
import torch
from logic.src.models.core.attention_model import AttentionModel
from torch import nn

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _model() -> AttentionModel:
    torch.manual_seed(0)
    return AttentionModel(
        embed_dim=16, hidden_dim=32, problem="ptp", n_encode_layers=1, n_heads=2, track_stats=True
    )


class _Wrapper(nn.Module):
    """Stands in for a LightningModule that holds the policy and a baseline with its own norms."""

    def __init__(self) -> None:
        super().__init__()
        self.policy = _model()
        self.baseline = nn.BatchNorm1d(4)


def _bn_keys(sd):
    return sorted(k for k in sd if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")))


def test_nested_load_keeps_policy_and_sibling_running_stats():
    src = _Wrapper()
    assert _bn_keys(src.policy.state_dict()), "the test model must track running stats"
    with torch.no_grad():
        for name, buf in src.named_buffers():
            if name.endswith(("running_mean", "running_var")):
                buf.fill_(3.0)
            elif name.endswith("num_batches_tracked"):
                buf.fill_(7)
    dst = _Wrapper()
    result = dst.load_state_dict(src.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for name, buf in dst.named_buffers():
        if name.endswith(("running_mean", "running_var")):
            assert torch.all(buf == 3.0), name
        elif name.endswith("num_batches_tracked"):
            assert int(buf) == 7, name


def test_top_level_load_still_drops_buffers_the_model_lacks():
    src = _model()
    dst = AttentionModel(
        embed_dim=16, hidden_dim=32, problem="ptp", n_encode_layers=1, n_heads=2, track_stats=False
    )
    dst.load_state_dict(src.state_dict(), strict=True)
    assert not _bn_keys(dst.state_dict())

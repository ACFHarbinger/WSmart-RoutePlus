import pytest
import torch
from logic.src.policies.vector.hgs_alns import VectorizedHGSALNS
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_vectorized_hgs_alns():
    batch_size = 2
    num_nodes = 10
    device = "cpu"

    # Mock data
    td = TensorDict(
        {
            "locs": torch.rand(batch_size, num_nodes - 1, 2, device=device),
            "depot": torch.rand(batch_size, 2, device=device),
            "waste": torch.rand(batch_size, num_nodes - 1, device=device),
            "capacity": torch.full((batch_size,), 10.0, device=device),
        },
        batch_size=[batch_size],
    )

    policy = VectorizedHGSALNS(
        env_name="ptp",
        time_limit=0.5,
        hgs={"mu": 4, "n_offspring": 4, "n_iterations_no_improvement": 2, "local_search_iterations": 5},
        alns={"max_iterations": 5},
        device=device,
    ).to(device)

    out = policy(td)

    assert "actions" in out
    assert "reward" in out
    assert "cost" in out
    assert out["actions"].shape[0] == batch_size
    assert out["reward"].shape[0] == batch_size

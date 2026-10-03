"""N2S must execute the declared action rather than silently repair it."""

import pytest
import torch
from logic.src.models.core.n2s.policy import execute_n2s_request_move


@pytest.mark.parametrize("action", [[1, 4], [1, 1, 0, 2], [1, 9, 0, 2], [1, 4, 99, 2], [1, 4, 3, 0]])
def test_invalid_request_action_rejected(action):
    solution = torch.arange(7).unsqueeze(0)
    original = solution.clone()
    with pytest.raises(ValueError):
        execute_n2s_request_move(solution, torch.tensor([action]))
    assert torch.equal(solution, original)

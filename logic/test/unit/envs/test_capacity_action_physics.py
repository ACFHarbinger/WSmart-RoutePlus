"""Capacity and action-shape physics for WCVRP, CVRPP, and SCWCVRP."""

from typing import Optional

import pytest
import torch
from logic.src.envs.routing.cvrpp import CVRPPEnv
from logic.src.envs.routing.swcvrp import SCWCVRPEnv
from logic.src.envs.routing.wcvrp import WCVRPEnv
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _wcvrp_td(
    *,
    waste: torch.Tensor,
    capacity: torch.Tensor,
    current_load: torch.Tensor,
    action: torch.Tensor,
    max_waste: Optional[torch.Tensor] = None,
    mandatory: Optional[torch.Tensor] = None,
) -> TensorDict:
    batch = waste.shape[0]
    n_nodes = waste.shape[-1]
    locs = torch.zeros(batch, n_nodes, 2)
    locs[:, 1, 0] = 1.0
    payload = {
        "locs": locs,
        "depot": torch.zeros(batch, 2),
        "waste": waste.clone(),
        "capacity": capacity,
        "current_load": current_load,
        "visited": torch.zeros(batch, n_nodes, dtype=torch.bool),
        "current_node": torch.zeros(batch, 1, dtype=torch.long),
        "tour": torch.zeros(batch, 0, dtype=torch.long),
        "tour_length": torch.zeros(batch),
        "collected_waste": torch.zeros(batch),
        "total_collected": torch.zeros(batch),
        "action": action,
        "i": torch.zeros(batch, dtype=torch.long),
    }
    if max_waste is not None:
        payload["max_waste"] = max_waste
    if mandatory is not None:
        payload["mandatory"] = mandatory
    return TensorDict(payload, batch_size=[batch])


def test_wcvrp_legal_visit_is_unchanged_when_waste_fits() -> None:
    """Capacity clamp is a no-op on the masked (legal) path."""
    env = WCVRPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = _wcvrp_td(
        waste=torch.tensor([[0.0, 0.4]]),
        capacity=torch.tensor([1.0]),
        current_load=torch.tensor([0.3]),
        action=torch.tensor([1]),
        max_waste=torch.tensor([1.0]),
    )

    next_td = env._step_instance(td.clone())

    assert next_td["current_load"].item() == pytest.approx(0.7)
    assert next_td["collected_waste"].item() == pytest.approx(0.4)
    assert next_td["waste"][0, 1].item() == pytest.approx(0.0)


def test_wcvrp_column_action_gathers_the_selected_node() -> None:
    """Decoder actions of shape [B, 1] must index the same node as [B]."""
    env = WCVRPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = _wcvrp_td(
        waste=torch.tensor([[0.0, 0.4]]),
        capacity=torch.tensor([1.0]),
        current_load=torch.zeros(1),
        action=torch.tensor([[1]]),
        max_waste=torch.tensor([1.0]),
    )

    next_td = env._step_instance(td)

    assert next_td["collected_waste"].item() == pytest.approx(0.4)
    assert int(next_td["current_node"].reshape(-1)[0].item()) == 1


def test_wcvrp_mandatory_customer_only_mask_aligns_depot() -> None:
    """A customer-only mandatory mask must not crash on the N vs N+1 depot column."""
    env = WCVRPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = _wcvrp_td(
        waste=torch.tensor([[0.0, 0.4, 0.2]]),
        capacity=torch.tensor([1.0]),
        current_load=torch.zeros(1),
        action=torch.tensor([1]),
        mandatory=torch.tensor([[True, False]]),
    )
    td["visited"][0, 0] = True

    mask = env._get_action_mask(td)

    assert mask.shape == torch.Size([1, 3])
    assert mask[0, 0].item() is False
    assert mask[0, 1].item() is True


def test_cvrpp_remaining_capacity_never_goes_negative() -> None:
    env = CVRPPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = TensorDict(
        {
            "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
            "depot": torch.zeros(1, 2),
            "waste": torch.tensor([[0.0, 0.8]]),
            "capacity": torch.tensor([0.5]),
            "remaining_capacity": torch.tensor([0.3]),
            "collected_waste": torch.zeros(1),
            "visited": torch.tensor([[True, False]]),
            "current_node": torch.zeros(1, 1, dtype=torch.long),
            "tour": torch.zeros(1, 0, dtype=torch.long),
            "tour_length": torch.zeros(1),
            "action": torch.tensor([1]),
        },
        batch_size=[1],
    )

    next_td = env._step_instance(td)

    assert next_td["remaining_capacity"].item() == pytest.approx(0.0)
    assert next_td["remaining_capacity"].item() >= 0.0


def test_scwcvrp_per_node_max_waste_counts_customer_overflows() -> None:
    env = SCWCVRPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = TensorDict(
        {
            "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
            "depot": torch.zeros(1, 2),
            "waste": torch.zeros(1, 3),
            "real_waste": torch.tensor([[0.0, 1.2, 0.4]]),
            "max_waste": torch.tensor([[1.0, 1.0, 1.0]]),
            "total_real_collected": torch.zeros(1),
            "tour_length": torch.zeros(1),
            "current_node": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=[1],
    )

    reward = env._get_reward(td)

    assert td["real_overflows"].item() == pytest.approx(1.0)
    assert reward.shape == torch.Size([1])


@pytest.mark.parametrize("per_customer", [False, True])
def test_scwcvrp_max_waste_without_depot_preserves_broadcasting(per_customer: bool) -> None:
    """Singleton limits and customer-only limits must not lose a column."""
    env = SCWCVRPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    limits = torch.tensor([[1.0], [0.3]])
    if per_customer:
        limits = torch.tensor([[1.0, 0.3], [2.0, 0.3]])
    td = TensorDict(
        {
            "locs": torch.zeros(2, 3, 2),
            "depot": torch.zeros(2, 2),
            "waste": torch.zeros(2, 3),
            "real_waste": torch.tensor([[0.0, 1.2, 0.4], [0.0, 1.2, 0.4]]),
            "max_waste": limits,
            "total_real_collected": torch.zeros(2),
            "tour_length": torch.zeros(2),
            "current_node": torch.zeros(2, 1, dtype=torch.long),
        },
        batch_size=[2],
    )

    reward = env._get_reward(td)

    expected = torch.tensor([2.0, 1.0] if per_customer else [1.0, 2.0])
    assert torch.equal(td["real_overflows"], expected)
    assert reward.shape == torch.Size([2])

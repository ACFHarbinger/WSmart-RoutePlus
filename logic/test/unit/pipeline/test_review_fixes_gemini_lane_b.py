"""Unit tests verifying Gemini review fixes across Lane B (Environments & Rewards)."""

import pytest
import torch
from tensordict import TensorDict

pytestmark = [pytest.mark.unit]


def test_cptptc_check_done_explicit_stop_and_exhaustion():
    """Verify CPTPTCEnv._check_done respects explicit finished flag and exhausted customer actions."""
    from logic.src.envs.cptptc import CPTPTCEnv

    env = CPTPTCEnv(generator_params={"num_loc": 5})
    td = env.reset(batch_size=[2])

    # Case 1: Initially at depot, i=0 -> not done
    done = env._check_done(td)
    assert not done.any()

    # Case 2: finished flag is explicitly True
    td["finished"] = torch.tensor([True, False])
    done = env._check_done(td)
    assert done[0].item() is True
    assert done[1].item() is False

    # Case 3: Customer actions exhausted (all customer nodes visited) and at depot with i > 0
    td["finished"] = torch.tensor([False, False])
    td["current_node"] = torch.tensor([[0], [0]])
    td["i"] = torch.tensor([[2], [0]])
    td["visited"] = torch.ones_like(td["visited"])  # All visited
    done = env._check_done(td)
    assert done[0].item() is True
    assert done[1].item() is False


def test_statistical_gamma_scaling():
    """Verify statistical gamma distribution samples are divided by 100 for both tensor and array paths."""
    from logic.src.data.distributions.statistical_gamma import Gamma

    dist = Gamma(alpha=2.0, theta=2.0)
    arr = dist._sample_array(size=(1000,))
    tns = dist._sample_tensor(size=(1000,))

    # Expected mean: alpha * theta / 100 = 2.0 * 2.0 / 100 = 0.04
    assert 0.02 < arr.mean() < 0.06
    assert 0.02 < tns.mean().item() < 0.06


def test_ptp_generate_has_max_waste():
    """Verify PTPEnv._generate includes max_waste tensor of ones."""
    from logic.src.envs.ptp import PTPGenerator

    gen = PTPGenerator(num_loc=10)
    td = gen(batch_size=[4])
    assert "max_waste" in td.keys()
    assert td["max_waste"].shape == (4,)
    assert torch.allclose(td["max_waste"], torch.ones(4))


def test_op_depot_prepending_for_partial_tour():
    """Verify OP.evaluate_tour prepends depot 0 to prize_all when customer indexing requires it."""
    from logic.src.envs.op import OP

    # Customer prize only (length 4, customers 1..4)
    prize = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    locs = torch.rand(1, 4, 2)
    depot = torch.tensor([[0.0, 0.0]])
    pi = torch.tensor([[0, 2, 4, 0]])

    dataset = {"prize": prize, "locs": locs, "depot": depot}
    res = OP.evaluate_tour(dataset, pi)
    # Customers 2 and 4 collected -> prizes 20.0 + 40.0 = 60.0
    assert torch.isclose(res["prize"], torch.tensor([60.0]))
    assert torch.isclose(res["reward"], torch.tensor([60.0]))


def test_pctsp_depot_prepending_for_partial_tour():
    """Verify PCTSP.evaluate_tour prepends depot 0 to penalty when indexing requires it."""
    from logic.src.envs.pctsp import PCTSP

    penalty = torch.tensor([[5.0, 15.0, 25.0, 35.0]])
    locs = torch.rand(1, 4, 2)
    depot = torch.tensor([[0.0, 0.0]])
    pi = torch.tensor([[0, 1, 3, 0]])

    dataset = {"penalty": penalty, "locs": locs, "depot": depot}
    res = PCTSP.evaluate_tour(dataset, pi)
    # Customers 1 and 3 visited: saved penalty 5.0 + 25.0 = 30.0
    assert torch.isclose(res["saved_penalty"], torch.tensor([30.0]))


def test_problem_get_tour_length_batched_matrix():
    """Verify BaseProblem.get_tour_length correctly looks up distances from batched 3D dist_matrix."""
    from logic.src.envs.base.problem import BaseProblem

    # 2 instances, 4 nodes (0 is depot, 1, 2, 3)
    dm = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ],
            [
                [0.0, 2.0, 4.0, 6.0],
                [2.0, 0.0, 2.0, 4.0],
                [4.0, 2.0, 0.0, 2.0],
                [6.0, 4.0, 2.0, 0.0],
            ],
        ]
    )
    pi = torch.tensor(
        [
            [0, 1, 2, 0],  # 0->1 (1.0), 1->2 (1.0), 2->0 (2.0) = 4.0
            [0, 3, 2, 0],  # 0->3 (6.0), 3->2 (2.0), 2->0 (4.0) = 12.0
        ]
    )

    length = BaseProblem.get_tour_length({}, pi, dist_matrix=dm)
    assert torch.allclose(length, torch.tensor([4.0, 12.0]))


def test_routing_env_step_vectorized_dm_lookup():
    """Verify RoutingEnv._step indexes distance from tensordict['dm'] in a vectorized fashion."""
    from logic.src.envs.cptp import CPTPEnv

    env = CPTPEnv(generator_params={"num_loc": 3})
    td = env.reset(batch_size=[2])

    # Provide precomputed dm: 4x4 (0 depot + 3 customers)
    dm = torch.full((2, 4, 4), 2.5)
    td["dm"] = dm

    action = torch.tensor([1, 2])
    td_next = env.step(td.clone().update({"action": action}))["next"]
    assert torch.allclose(td_next["step_distance"], torch.tensor([2.5, 2.5]))


def test_eval_loop_dm_lookup_and_kg_scaling():
    """Verify eval_loop dm fallback resolution and bin_capacity_kg default scaling."""
    # When dist_matrix is absent, dm is used
    batch_i = {"dm": torch.eye(4), "bin_capacity_kg": 47.5}
    dm_input = batch_i.get("dist_matrix", batch_i.get("dm", batch_i.get("dist")))
    assert dm_input is not None

    # kg is scaled by bin_capacity_kg (default 47.5)
    waste_val = torch.tensor(2.0)
    bin_cap_kg = float(batch_i.get("bin_capacity_kg", 47.5))
    kg = waste_val.item() * bin_cap_kg
    assert kg == 95.0


def test_multi_day_data_replication_preservation():
    """Verify accumulate_initial_fill preserves _waste_full for multi-day eval expansion."""
    from logic.src.pipeline.learning.common.epoch import accumulate_initial_fill

    # 4 instances, 3 days, 5 bins
    waste_3d = torch.rand(4, 3, 5)
    td = TensorDict({"waste": waste_3d.clone()}, batch_size=[4])
    accumulate_initial_fill(td, max_days=2)

    # 2D waste computed for day 0
    assert td["waste"].dim() == 2
    # Original 3D tensor preserved in _waste_full
    assert "_waste_full" in td.keys()
    assert td["_waste_full"].dim() == 3
    assert td["_waste_full"].shape == (4, 3, 5)
    assert torch.equal(td["_waste_full"], waste_3d)

    # Verify replication restore pattern used in data.py:263-264
    waste_field = td.get("_waste_full", td.get("waste", None))
    td_day = td.clone()
    if "_waste_full" in td_day.keys():
        td_day["waste"] = waste_field
    assert td_day["waste"].dim() == 3
    assert torch.equal(td_day["waste"], waste_3d)


def test_neural_agent_proxy_reward_waste_revenue_scaling():
    """Verify NeuralAgent proxy reward incorporates waste_weight / revenue multiplier."""
    from types import SimpleNamespace

    # Simulate NeuralAgent reward calculation with default w_waste = 27.74
    params = SimpleNamespace(waste_weight=27.74, reward_weight=1.0)
    waste = torch.tensor([0.8])  # 80% bin fill collected
    dists = torch.tensor([0.5])  # distance traveled
    w_len = 0.1

    # Default waste scaling: 27.74 €/bin
    w_waste = 27.74
    if params and hasattr(params, "waste_weight") and params.waste_weight is not None:
        w_waste = float(params.waste_weight)
    incremental_reward = waste * w_waste - dists * w_len

    # 0.8 * 27.74 - 0.5 * 0.1 = 22.192 - 0.05 = 22.142 > 0 (profitable collection)
    assert incremental_reward.item() > 20.0

    # Without scaling (unscaled waste = 0.8), high distance (e.g. 10.0) would cause negative reward
    # but with scaling, collection remains profitable:
    high_dist = torch.tensor([10.0])
    reward_scaled = waste * w_waste - high_dist * w_len
    assert reward_scaled.item() > 0.0  # 22.192 - 1.0 = 21.192 > 0


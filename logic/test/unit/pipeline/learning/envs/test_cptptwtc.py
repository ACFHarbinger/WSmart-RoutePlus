"""CPTPTWTC: CPTPTC's capacity and shift budget plus bin time windows."""

import pytest
import torch
from logic.src.envs import CPTPTWTC, PROBLEMS, get_env
from logic.src.envs.cptptwtc import CPTPTWTCEnv, CPTPTWTCGenerator, time_window_violations

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _instance(early, late, locs, day=10.0):
    """One instance: depot at the origin, bins at ``locs``, 60 km/h, no service time."""
    td = CPTPTWTCGenerator(num_loc=len(locs))(batch_size=[1])
    td["locs"] = torch.tensor([locs], dtype=torch.float32)
    td["depot"] = torch.zeros(1, 2)
    td["waste"] = torch.full((1, len(locs)), 1.0)
    td["capacity"] = torch.tensor([100.0])  # loads never bind: only the windows decide
    td["tw_early"] = torch.tensor([early], dtype=torch.float32)
    td["tw_late"] = torch.tensor([late], dtype=torch.float32)
    td["day_hours"] = torch.tensor([day])
    td["avg_speed_kmh"] = torch.tensor([60.0])
    td["service_time_h"] = torch.tensor([0.0])
    td["shift_hours"] = torch.tensor([24.0])
    return td


def _rollout(env, td, seed=0, steps=80):
    g = torch.Generator().manual_seed(seed)
    actions = []
    for _ in range(steps):
        a = torch.multinomial(td["action_mask"].float(), 1, generator=g).squeeze(-1)
        td.set("action", a)
        td = env.step(td)["next"]
        actions.append(a)
        if td["done"].all():
            break
    return td, torch.stack(actions, 1)


def test_generator_adds_windows_inside_the_day():
    td = CPTPTWTCGenerator(num_loc=20, day_hours=8.0, tw_min_width_h=1.0, tw_max_width_h=3.0)(batch_size=[32])
    width = td["tw_late"] - td["tw_early"]
    assert td["tw_early"].shape == (32, 20)
    assert bool((td["tw_early"] >= 0).all()) and bool((td["tw_late"] <= 8.0 + 1e-6).all())
    assert bool((width >= 1.0 - 1e-6).all()) and bool((width <= 3.0 + 1e-6).all())


def test_generator_rejects_impossible_windows():
    with pytest.raises(ValueError):
        CPTPTWTCGenerator(num_loc=5, day_hours=2.0, tw_min_width_h=3.0, tw_max_width_h=4.0)


def test_early_arrival_waits_for_the_window_to_open():
    env = CPTPTWTCEnv(num_loc=1)
    td = env.reset(_instance(early=[2.0], late=[5.0], locs=[[0.6, 0.8]]))  # 1 km away: 1 min at 60 km/h
    td.set("action", torch.tensor([1]))
    td = env.step(td)["next"]
    assert td["current_time"].item() == pytest.approx(2.0)


def test_a_closed_window_is_masked():
    env = CPTPTWTCEnv(num_loc=2)
    # Bin 1 closes before the vehicle can get there; bin 2 is open all morning.
    td = env.reset(_instance(early=[0.0, 0.0], late=[0.001, 5.0], locs=[[3.0, 4.0], [0.3, 0.4]]))
    assert td["action_mask"][0].tolist() == [True, False, True]


def test_masked_rollouts_never_violate_a_window():
    for seed in range(3):
        env = get_env("cptptwtc", num_loc=12)
        instances = env.generator(batch_size=[16])
        td, pi = _rollout(env, env.reset(instances.clone()), seed=seed)
        assert int(time_window_violations(instances, pi).sum()) == 0
        assert int(CPTPTWTC.evaluate_tour(td, pi)["tw_violations"].sum()) == 0


def test_evaluate_tour_reports_and_rejects_a_late_visit():
    td = _instance(early=[0.0], late=[0.001], locs=[[3.0, 4.0]])
    pi = torch.tensor([[1, 0]])
    assert CPTPTWTC.evaluate_tour(td, pi)["tw_violations"].tolist() == [1]
    with pytest.raises(AssertionError, match="time window"):
        CPTPTWTC.evaluate_tour(td, pi, check_feasibility=True)


def test_problem_is_registered_with_its_model_components():
    from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY
    from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
    from logic.src.models.subnets.embeddings.cptptwtc import CPTPTWTCInitEmbedding
    from logic.src.utils.model.problem_factory import load_problem

    assert PROBLEMS.get("cptptwtc").get_env_class() is CPTPTWTCEnv
    assert load_problem("cptptwtc") is CPTPTWTC
    assert INIT_EMBEDDING_REGISTRY["cptptwtc"] is CPTPTWTCInitEmbedding
    assert CONTEXT_EMBEDDING_REGISTRY["cptptwtc"] is CONTEXT_EMBEDDING_REGISTRY["cptp"]


def test_attention_model_decodes_window_feasible_tours():
    from logic.src.models.core.attention_model.policy import AttentionModelPolicy

    torch.manual_seed(0)
    env = get_env("cptptwtc", num_loc=10)
    instances = env.generator(batch_size=[8])
    policy = AttentionModelPolicy(env_name="cptptwtc", embed_dim=32, hidden_dim=32, n_encode_layers=1)
    out = policy(env.reset(instances.clone()), env, strategy="greedy")
    assert int(time_window_violations(instances, out["actions"]).sum()) == 0


def test_window_sampler_is_seeded_and_bounded():
    from logic.src.data.time.node_time_windows import sample_node_time_windows

    a = sample_node_time_windows((3, 7), generator=torch.Generator().manual_seed(1))
    b = sample_node_time_windows((3, 7), generator=torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert bool((a[0] >= 0).all()) and bool((a[1] <= 10.0 + 1e-6).all())


def test_generated_datasets_carry_the_time_windows():
    from logic.src.data.generators.builders import VRPInstanceBuilder

    builder = VRPInstanceBuilder(device="cpu")
    builder.set_dataset_size(4).set_problem_size(10).set_waste_type("plastic").set_distribution("gamma3")
    builder.set_area("riomaior").set_focus_graph(None, 0).set_problem_name("cptptwtc").set_num_days(1).set_seed(3)
    td = builder.build_td()
    assert td["tw_early"].shape == (4, 10) and td["tw_late"].shape == (4, 10)
    assert bool((td["tw_late"] > td["tw_early"]).all())

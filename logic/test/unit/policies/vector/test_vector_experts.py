"""Contract tests of the vectorised experts against the simulator policy configs."""

import dataclasses

import pytest
import torch
from logic.src.configs.policies import (
    ALNSConfig,
    HGSALNSConfig,
    HGSConfig,
    HVPLConfig,
    ILSConfig,
    KSparseACOConfig,
)
from logic.src.configs.policies.other.route_improvement import RandomLocalSearchPostConfig
from logic.src.configs.policies.profitable_nn import ProfitableNNConfig
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.factory import EXPERT_TARGETS, create_expert_policy, load_expert_policy_cls
from logic.src.policies.vectorized.routing.shared.vrpp import parse_instance, profit_split
from tensordict import TensorDict

pytestmark = [pytest.mark.unit]

SMALL = {
    "ALNSConfig": lambda: ALNSConfig(max_iterations=30, time_limit=5.0),
    "HGSConfig": lambda: HGSConfig(
        mu=4, n_offspring=4, n_iterations_no_improvement=5, local_search_iterations=5, time_limit=5.0
    ),
    "HGSALNSConfig": lambda: HGSALNSConfig(
        hgs=HGSConfig(mu=4, n_offspring=4, n_iterations_no_improvement=3, local_search_iterations=5),
        alns=ALNSConfig(max_iterations=5),
    ),
    "HVPLConfig": lambda: HVPLConfig(
        n_teams=3, max_iterations=2, aco_init_iterations=2, alns_params=ALNSConfig(max_iterations=5), time_limit=5.0
    ),
    "ILSConfig": lambda: ILSConfig(n_restarts=3, inner_iterations=3, time_limit=5.0),
    "KSparseACOConfig": lambda: KSparseACOConfig(n_ants=4, max_iterations=3, local_search_iterations=5, k_sparse=5),
    "RandomLocalSearchPostConfig": lambda: RandomLocalSearchPostConfig(iterations=20),
    "ProfitableNNConfig": lambda: ProfitableNNConfig(),
}


def _td(B=3, N=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDict(
        {
            "locs": torch.rand(B, N, 2, generator=g),
            "depot": torch.rand(B, 2, generator=g),
            "waste": torch.rand(B, N, generator=g) * 0.6,
            "capacity": torch.ones(B),
        },
        batch_size=[B],
    )


def test_every_expert_is_registered():
    assert set(SMALL) == set(EXPERT_TARGETS)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_expert_takes_its_simulator_config(name):
    cfg = SMALL[name]()
    policy = create_expert_policy(cfg, env_name="cptp", seed=1, device="cpu")
    assert type(policy).__name__ == EXPERT_TARGETS[name][1]
    if hasattr(policy, "cfg"):
        assert type(policy.cfg) is type(cfg)
        assert dataclasses.asdict(policy.cfg) == dataclasses.asdict(cfg)


@pytest.mark.parametrize("name", sorted(n for n in SMALL if n != "ProfitableNNConfig"))
def test_expert_rejects_unknown_parameters(name):
    cls = load_expert_policy_cls(SMALL[name]())
    with pytest.raises(ValueError):
        cls("cptp", population_size=10)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_expert_solutions_are_feasible(name):
    td = _td()
    policy = create_expert_policy(SMALL[name](), env_name="cptp", seed=3, device="cpu")
    out = policy(td.clone())
    inst = parse_instance(td)
    B = td.batch_size[0]
    assert out["actions"].shape[0] == B
    assert out["reward"].shape == (B,)
    for b in range(B):
        bins = out["actions"][b][out["actions"][b] != 0].tolist()
        assert len(bins) == len(set(bins))
        # The simulator's SweepLocalSearch swap_star uses a soft capacity penalty
        # (empty top-insertion cache), so RLS may legitimately overload a route.
        if "routes" in out and name != "RandomLocalSearchPostConfig":
            for route in out["routes"][b]:
                assert float(inst.waste[b, route].sum()) <= float(inst.capacity[b]) + 1e-5


@pytest.mark.parametrize("name", ["ALNSConfig", "ILSConfig", "RandomLocalSearchPostConfig", "KSparseACOConfig"])
def test_expert_is_seed_deterministic(name):
    td = _td()
    a = create_expert_policy(SMALL[name](), env_name="cptp", seed=5, device="cpu")(td.clone())
    b = create_expert_policy(SMALL[name](), env_name="cptp", seed=5, device="cpu")(td.clone())
    assert torch.equal(a["actions"], b["actions"])


def test_route_sequence_roundtrip_and_profit():
    td = _td(B=2, N=8)
    inst = parse_instance(td)
    S = rs.from_route_lists([[[1, 2], [3]], []], inst.n_bins, inst.device)
    assert rs.to_route_lists(S) == [[[1, 2], [3]], []]
    d = inst.dist[0]
    expected = float(
        inst.R * inst.waste[0, [1, 2, 3]].sum() - inst.C * (d[0, 1] + d[1, 2] + d[2, 0] + d[0, 3] + d[3, 0])
    )
    assert abs(float(rs.seq_profit(S, inst)[0]) - expected) < 1e-4


def test_profit_split_skips_unprofitable_bins():
    td = _td(B=1, N=6)
    td["waste"][:] = 0.0
    inst = parse_instance(td)
    tours = torch.arange(1, 7).unsqueeze(0)
    routes, profit, _ = profit_split(tours, inst, allow_skip=True)
    assert routes == [[]]
    assert float(profit[0]) == 0.0

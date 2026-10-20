"""Surviving packing defects from the 2026-10-20 Cursor review."""

from types import SimpleNamespace

import numpy as np
import pytest

from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.pipeline.simulations.actions.collection import CollectAction
from logic.src.pipeline.simulations.actions.node_selection import PackingAction
from logic.src.policies.packing.combined import CombinedSelection
from logic.src.policies.packing.deadline import DeadlineDrivenSelection
from logic.src.policies.packing.dispatcher_thompson import ThompsonDispatcher
from logic.src.policies.packing.fractional_knapsack import FractionalKnapsackSelection
from logic.src.policies.packing.kmeans_sector import KMeansGeographicSectorSelection
from logic.src.policies.packing.mip_knapsack import MIPKnapsackSelection
from logic.src.policies.packing.rollout import RolloutSelection
from logic.src.policies.packing.submodular_greedy import SubmodularGreedySelection
from logic.src.policies.packing.supermodular_greedy import SupermodularGreedySelection
from logic.src.policies.packing.whittle import WhittleIndexSelection

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _ctx(**kwargs) -> PackingContext:
    n = len(kwargs.get("current_fill", []))
    base = dict(
        bin_ids=np.arange(n, dtype=np.int32),
        current_fill=np.zeros(n),
        max_fill=100.0,
        bin_volume=1.0,
        bin_density=1.0,
        revenue_kg=1.0,
        vehicle_capacity=100.0,
        n_vehicles=1,
        cost_per_km=1.0,
    )
    base.update(kwargs)
    return PackingContext(**base)


def _distance(n_bins: int, value: float = 1.0) -> np.ndarray:
    dm = np.full((n_bins + 1, n_bins + 1), value, dtype=float)
    np.fill_diagonal(dm, 0.0)
    return dm


def test_submodular_greedy_returns_a_selection() -> None:
    fills = np.array([80.0, 40.0, 10.0, 95.0])
    selected, _ = SubmodularGreedySelection().pack(_ctx(current_fill=fills, distance_matrix=_distance(4)))
    assert isinstance(selected, list)


def test_supermodular_greedy_survives_lazy_reevaluation() -> None:
    fills = np.array([80.0, 40.0, 10.0, 95.0])
    selected, _ = SupermodularGreedySelection().pack(
        _ctx(current_fill=fills, distance_matrix=_distance(4), modular_alpha=0.01)
    )
    assert isinstance(selected, list)
    assert selected


def test_fractional_knapsack_keeps_the_higher_net_profit_set() -> None:
    # A 100 kg bin, so fill percent is kg and revenue is €1/kg. Revenues are
    # 35, 15 and 15; round-trip nets are 35 - 30 = 5 and 15 - 0.2 = 14.8 each.
    # The big bin's revenue (35) beats the two small bins' (30), but the two
    # small bins are the better feasible set by net profit (29.6 against 5).
    fills = np.array([35.0, 15.0, 15.0])
    dm = _distance(3, 0.1)
    dm[0, 1] = dm[1, 0] = 15.0
    dm[1, 2] = dm[2, 1] = dm[1, 3] = dm[3, 1] = 15.0
    selected, _ = FractionalKnapsackSelection().pack(
        _ctx(current_fill=fills, distance_matrix=dm, bin_volume=100.0, vehicle_capacity=40.0, cost_per_km=1.0)
    )
    assert sorted(selected) == [2, 3]


def test_combined_forwards_child_parameters() -> None:
    fills = np.array([80.0, 40.0, 10.0, 95.0])
    selected, _ = CombinedSelection(
        strategies=[
            {"name": "last_minute", "params": {"threshold": 90.0}},
            {"name": "revenue", "params": {"revenue_threshold": 100000.0, "revenue_kg": 0.01}},
        ],
        logic="and",
    ).pack(_ctx(current_fill=fills, bin_volume=2.5, bin_density=19.0, revenue_kg=0.01, threshold=0.0))
    assert selected == []


def test_packing_action_uses_bin_expenses_as_cost_per_km() -> None:
    seen = {}

    def _pack(self, context):
        del self
        seen["cost_per_km"] = context.cost_per_km
        return [], SearchContext.initialize()

    bins = SimpleNamespace(
        c=np.array([50.0]),
        means=np.array([1.0]),
        std=np.array([0.0]),
        n=1,
        revenue=1.0,
        density=1.0,
        volume=1.0,
        expenses=1.0,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FractionalKnapsackSelection, "pack", _pack)
        PackingAction().execute(
            {
                "bins": bins,
                "config": {"packing": {"fractional_knapsack": {}}},
                "distance_matrix": _distance(1),
            }
        )
    assert seen["cost_per_km"] == 1.0


def test_mip_knapsack_sees_a_scenario_tree_for_a_nearly_full_bin() -> None:
    bins = SimpleNamespace(
        c=np.array([99.0, 100.0]),
        means=np.array([5.0, 5.0]),
        std=np.array([0.0, 0.0]),
        n=2,
        revenue=1.0,
        density=1.0,
        volume=1.0,
        expenses=1.0,
    )
    ctx = {
        "bins": bins,
        "config": {"packing": {"mip_knapsack": {}}, "horizon": 2},
        "distance_matrix": _distance(2),
        "day": 0,
        "vehicle_capacity": 10000.0,
    }
    PackingAction().execute(ctx)
    assert 1 in ctx["mandatory"]


def test_whittle_capacity_ignores_empty_bins() -> None:
    # One full Rio Maior bin (47.5 kg) and 19 empty ones; the vehicle holds one full bin.
    fills = np.zeros(20)
    fills[0] = 100.0
    selected, _ = WhittleIndexSelection().pack(
        _ctx(
            current_fill=fills,
            accumulation_rates=np.full(20, 10.0),
            vehicle_capacity=47.5,
            n_vehicles=1,
            bin_volume=2.5,
            bin_density=19.0,
        )
    )
    assert selected == [1]


def test_thompson_respects_seed_and_records_the_day_reward() -> None:
    ThompsonDispatcher._shared_state = {}
    context = _ctx(
        current_fill=np.array([95.0, 10.0]),
        seed=1,
        dispatcher_candidate_strategies=["last_minute"],
        threshold=90.0,
    )
    first, first_ctx = ThompsonDispatcher().pack(context)
    second, _ = ThompsonDispatcher().pack(context)
    assert first == second
    assert first_ctx.selection_metrics["winner"] == "last_minute"

    bins = SimpleNamespace(collect=lambda tour, km: (1.0, 1.0, 1, 4.0))
    CollectAction().execute(
        {
            "bins": bins,
            "tour": [0, 1, 0],
            "distance_matrix": _distance(1),
            "search_context": first_ctx,
        }
    )
    alpha, beta = ThompsonDispatcher._shared_state["last_minute"]
    assert alpha == 2.0
    assert beta == 1.0
    ThompsonDispatcher._shared_state = {}


def test_kmeans_reads_day_coords() -> None:
    bins = SimpleNamespace(
        c=np.array([10.0, 20.0, 30.0, 40.0]),
        means=np.zeros(4),
        std=np.zeros(4),
        n=4,
        revenue=1.0,
        density=1.0,
        volume=1.0,
        expenses=1.0,
    )
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ctx = {
        "bins": bins,
        "coords": coords,
        "config": {"packing": {"kmeans_sector": {"threshold": 1}}},
        "day": 1,
    }
    PackingAction().execute(ctx)
    assert ctx["mandatory"] == [1, 2, 3, 4]
    direct, _ = KMeansGeographicSectorSelection().pack(
        _ctx(current_fill=bins.c, coordinates=coords, threshold=1, current_day=1)
    )
    assert direct == [1, 2, 3, 4]


def test_deadline_does_not_collect_a_bin_still_under_capacity() -> None:
    selected, _ = DeadlineDrivenSelection().pack(
        _ctx(current_fill=np.array([71.0]), accumulation_rates=np.array([10.0]), threshold=2.0)
    )
    assert selected == []


def test_rollout_monte_carlo_follows_the_context_seed() -> None:
    rng = np.random.default_rng(0)
    n = 12
    context = _ctx(
        current_fill=rng.uniform(60.0, 99.0, n),
        accumulation_rates=rng.uniform(2.0, 15.0, n),
        std_deviations=np.full(n, 8.0),
        threshold=90.0,
        rollout_n_scenarios=10,
        rollout_horizon=3,
        seed=1,
        distance_matrix=np.full((n + 1, n + 1), 0.3),
        bin_volume=2.5,
        bin_density=19.0,
        revenue_kg=0.6,
        vehicle_capacity=3500.0,
    )
    results = set()
    for global_seed in range(15):
        np.random.seed(global_seed)  # the process generator must not matter
        results.add(tuple(RolloutSelection().pack(context)[0]))
    assert len(results) == 1


def test_mip_pack_uses_an_explicit_tree() -> None:
    bins = SimpleNamespace(c=np.array([99.0]), means=np.array([5.0]), std=np.array([0.0]))
    from logic.src.pipeline.simulations.actions.node_selection import _packing_scenario_tree

    tree = _packing_scenario_tree(
        SimpleNamespace(bins=bins, scenario_tree=None, seed=1),
        {"horizon": 2},
        "mip_knapsack",
    )
    selected, _ = MIPKnapsackSelection().pack(
        _ctx(
            current_fill=np.array([99.0]),
            distance_matrix=_distance(1),
            scenario_tree=tree,
            vehicle_capacity=10.0,
        )
    )
    assert selected == [1]

"""Surviving route-improvement defects from the 2026-10-20 Cursor review."""

from types import SimpleNamespace

import numpy as np
import pytest
from logic.src.configs.policies.routing.meta_heuristics.ts import TSConfig
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from logic.src.policies.routing.meta_heuristics.tabu_search.solver import TSSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_tabu_selects_least_rejected_when_all_rejected() -> None:
    """When every neighbour is rejected, Tabu installs the highest-profit one, not candidates[0]."""
    # Minimal setup that forces all-rejected path
    dist = np.eye(3) * 0 + 1.0
    wastes = {1: 10.0, 2: 20.0}
    params = TSConfig(seed=42, max_iterations=1, acceptance_criterion={"method": "oi"})
    solver = TSSolver(dist, wastes, capacity=100.0, R=1.0, C=1.0, params=params)

    # ``solve`` builds the criterion; here every neighbour is rejected.
    solver.acceptance = SimpleNamespace(accept=lambda **kwargs: (False, {}))
    solver.iteration = 0

    candidates = [
        ([[1]], ("swap", (1, 2))),
        ([[2]], ("swap", (2, 1))),
    ]
    # Make second better
    solver._evaluate = lambda routes: 100.0 if routes == [[2]] else 50.0

    best, profit, desc = solver._select_best_candidate(candidates, current_profit=0.0, best_profit=0.0)
    assert best == [[2]]
    assert profit == 100.0


# ---------------------------------------------------------------------------
# Helpers for the operator tests
# ---------------------------------------------------------------------------


class _FakeLS:
    """Just enough of LocalSearch for the relocate and SWAP* operators."""

    def __init__(self, dist, routes, waste=None, capacity=1e9):
        self.d = np.asarray(dist, dtype=float)
        self.routes = [list(r) for r in routes]
        self.waste = waste or {}
        self.Q = capacity
        self.C = 1.0
        self.penalty_capacity = 1.0
        self.top_insertions = {}

    def get_dist(self, i, j):
        return float(self.d[i, j])

    def _get_load_cached(self, r):
        return sum(self.waste.get(n, 0) for n in self.routes[r])

    def _update_map(self, _routes):
        pass


def _line(points):
    pts = np.asarray(points, dtype=float)
    return np.abs(pts[:, None] - pts[None, :])


def _tour_cost(dm, tour):
    return float(sum(dm[a, b] for a, b in zip(tour[:-1], tour[1:])))


# ---------------------------------------------------------------------------
# Simulated annealing and ruin-recreate: tour cost into a maximising rule
# ---------------------------------------------------------------------------


def test_simulated_annealing_hands_the_criterion_negated_lengths(monkeypatch) -> None:
    from logic.src.policies.acceptance_criteria import boltzmann_metropolis_criterion as sa_mod
    from logic.src.policies.route_improvement.simulated_annealing import SimulatedAnnealingRouteImprover

    seen = []
    original = sa_mod.BoltzmannAcceptance.accept

    def spy(self, current_obj, candidate_obj, **kwargs):
        seen.append((current_obj, candidate_obj))
        return original(self, current_obj, candidate_obj, **kwargs)

    monkeypatch.setattr(sa_mod.BoltzmannAcceptance, "accept", spy)
    dm = _line([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    tour = [0, 1, 2, 3, 4, 5, 0]  # optimal on a line, so every move is longer
    refined, _ = SimulatedAnnealingRouteImprover().process(
        tour, distance_matrix=dm, sa_iterations=200, sa_t_init=1e-12, sa_cooling=1.0, seed=3
    )
    assert seen
    # BoltzmannAcceptance maximises: a longer tour must reach it as a lower value.
    assert all(cur <= 0 and cand <= cur + 1e-9 for cur, cand in seen)
    assert _tour_cost(dm, refined) == pytest.approx(10.0)


def test_ruin_recreate_sa_mode_negates_tour_cost(monkeypatch) -> None:
    from logic.src.policies.acceptance_criteria import boltzmann_metropolis_criterion as sa_mod

    seen = []
    original = sa_mod.BoltzmannAcceptance.accept

    def spy(self, current_obj, candidate_obj, **kwargs):
        seen.append((current_obj, candidate_obj))
        return original(self, current_obj, candidate_obj, **kwargs)

    monkeypatch.setattr(sa_mod.BoltzmannAcceptance, "accept", spy)
    from logic.src.policies.route_improvement.ruin_recreate import RuinRecreateRouteImprover

    dm = _line([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    RuinRecreateRouteImprover().process(
        [0, 1, 2, 3, 4, 5, 0], distance_matrix=dm, lns_acceptance="sa", lns_iterations=20, seed=1
    )
    assert seen, "SA mode never reached the criterion"
    # Objectives are negated distances.
    assert all(cur <= 0 and cand <= 0 for cur, cand in seen)


# ---------------------------------------------------------------------------
# Steepest 2-opt on a directed matrix
# ---------------------------------------------------------------------------


def test_steepest_two_opt_does_not_lengthen_a_directed_tour() -> None:
    from logic.src.policies.helpers.operators.improvement_descent.steepest_two_opt import two_opt_steepest

    rng = np.random.default_rng(0)
    for _ in range(25):
        n = 7
        dm = rng.uniform(1.0, 10.0, size=(n, n))
        np.fill_diagonal(dm, 0.0)
        route = list(rng.permutation(np.arange(1, n)))
        before = _tour_cost(dm, [0] + route + [0])
        improved = two_opt_steepest([route], dm, {}, 1e9)
        after = _tour_cost(dm, [0] + improved[0] + [0])
        assert after <= before + 1e-9


# ---------------------------------------------------------------------------
# Relocate after the depot
# ---------------------------------------------------------------------------


def test_relocate_sweep_tries_the_slot_after_the_depot() -> None:
    from logic.src.policies.helpers.operators.intra_route_local_search.relocate import move_relocate

    dm = _line([0.0, 1.0, 5.0, 6.0])
    # Route 0 holds node 2 far from node 1; node 1 is best placed first.
    ls = _FakeLS(dm, [[2, 1, 3]])
    assert move_relocate(ls, 1, 0, 0, 1, 0, -1)
    assert ls.routes[0] == [1, 2, 3]


def test_sweep_local_search_relocate_loop_starts_at_the_depot() -> None:
    import inspect

    from logic.src.policies.helpers.local_search import local_search_sweeps

    source = inspect.getsource(local_search_sweeps.SweepLocalSearch)
    assert "range(-1, len(route_v))" in source


# ---------------------------------------------------------------------------
# SWAP* with an empty Top-3 cache
# ---------------------------------------------------------------------------


def test_swap_star_prices_every_slot_on_a_cache_miss() -> None:
    from logic.src.policies.helpers.operators.inter_route_local_search.swap_star import move_swap_star

    # Found by search: swapping 3 and 6 shortens the plan from 85.7 to 75.1,
    # but only when 6 goes to the front of route A, not into the slot 3 left.
    pts = np.array([[0, 0], [7, 3], [0, -5], [-4, -10], [-9, -10], [-7, 7], [3, 9]], dtype=float)
    dm = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    ls = _FakeLS(dm, [[1, 2, 3], [4, 5, 6]])
    before = sum(_tour_cost(dm, [0] + r + [0]) for r in ls.routes)
    assert move_swap_star(ls, 3, 6, 0, 2, 1, 2)
    after = sum(_tour_cost(dm, [0] + r + [0]) for r in ls.routes)
    assert after < before - 1.0


# ---------------------------------------------------------------------------
# RouteImprovementAction keeps only better tours
# ---------------------------------------------------------------------------


class _FixedImprover:
    def __init__(self, tour):
        self._tour = tour

    def process(self, tour, **kwargs):
        return list(self._tour), {"algorithm": "fixed"}


class _Ctx:
    def __init__(self, tour, dm):
        self.tour = tour
        self.distance_matrix = dm
        self.cost = _tour_cost(dm, tour)
        self.search_context = None

    def as_kwargs(self):
        return {"tour": self.tour, "distance_matrix": self.distance_matrix}


def test_route_improvement_action_rejects_a_longer_reordering() -> None:
    from logic.src.pipeline.simulations.actions.route_improvement import RouteImprovementAction

    dm = _line([0.0, 1.0, 2.0, 3.0])
    ctx = _Ctx([0, 1, 2, 3, 0], dm)
    RouteImprovementAction()._apply_processors([_FixedImprover([0, 2, 1, 3, 0])], ctx)
    assert ctx.tour == [0, 1, 2, 3, 0]


def test_route_improvement_action_keeps_an_insertion_that_adds_a_bin() -> None:
    from logic.src.pipeline.simulations.actions.route_improvement import RouteImprovementAction

    dm = _line([0.0, 1.0, 2.0, 3.0])
    ctx = _Ctx([0, 1, 2, 0], dm)
    RouteImprovementAction()._apply_processors([_FixedImprover([0, 1, 2, 3, 0])], ctx)
    assert ctx.tour == [0, 1, 2, 3, 0]
    assert ctx.cost == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Packaged acceptance files build the criterion they name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["gd", "rrt", "nlgd", "ta", "lahc", "schc", "oba", "svns", "gt_sa", "ema"])
def test_packaged_acceptance_file_builds_and_runs(stem: str) -> None:
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[5] / "logic" / "configs" / "policies" / "other" / "acceptance_criteria" / f"ac_{stem}.yaml"
    section = next(iter(yaml.safe_load(path.read_text()).values()))
    # The live path (HGS, PSOMA, ALNS) passes the params mapping as ``config``.
    criterion = AcceptanceCriterionFactory.create(name=section["method"], config=section.get("params"))
    criterion.setup(100.0)
    accepted, _ = criterion.accept(current_obj=100.0, candidate_obj=110.0)
    criterion.step(current_obj=100.0, candidate_obj=110.0, accepted=accepted)
    assert accepted, f"{stem}: an improving move must be accepted"

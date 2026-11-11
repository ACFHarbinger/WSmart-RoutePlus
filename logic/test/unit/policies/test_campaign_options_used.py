"""Options of the campaign's routing algorithms that used to be ignored now act (and their defaults reproduce the runs)."""

import dataclasses
from pathlib import Path
from typing import List

import numpy as np
import pytest
from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bpc import BPCConfig
from logic.src.configs.policies.routing.hyper_heuristics.aco_hh import HyperHeuristicACOConfig
from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.configs.policies.routing.meta_heuristics.psoma import PSOMAConfig
from logic.src.configs.policies.routing.other_algorithms.cvrp import CVRPConfig
from logic.src.policies.routing.base.sub_problem import SubProblem, build_subproblem
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
    HyperHeuristicACO,
)
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver import (
    PSOMASolver,
)


def _problem(n: int = 12, seed: int = 0, revenue: float = 0.3) -> SubProblem:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n + 1, 2))
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    fill = rng.uniform(5, 100, n)
    mandatory = [i + 1 for i in range(n) if fill[i] >= 80] or [1]
    return build_subproblem(dist, {i + 1: float(fill[i]) for i in range(n)}, 300.0, revenue, 1.0, mandatory)


def test_defaults_reproduce_the_reported_runs() -> None:
    assert HyperHeuristicACOConfig().local_search is False and HyperHeuristicACOConfig().elitist_weight == 1.0
    assert HGSConfig().mutation_rate == 1.0 and HGSConfig().profit_aware_operators is True
    assert PSOMAConfig().profit_aware_operators is True
    bpc = BPCConfig()
    assert bpc.enable_dssr is True and bpc.dssr_max_iters == 0 and bpc.enable_reduced_cost_arc_fixing is True
    assert CVRPConfig().cache is False


def test_aco_hh_elitist_weight_scales_the_best_journey() -> None:
    solver = HyperHeuristicACO(_problem(), HyperHeuristicACOConfig(seed=1, elitist_weight=3.0), [[1]])
    n = solver.n_operators
    solver.tau = np.zeros((n + 1, n + 1)) if solver.tau.shape[0] > n else np.zeros_like(solver.tau)
    op = solver.operator_names
    # (routes, start_obj, final_obj, sequence, ant_idx, start_idx): ant 0 improves by 10, ant 1 by 2
    solver._deposit_pheromones([([[1]], 20.0, 10.0, [op[0]], 0, 1), ([[1]], 20.0, 18.0, [op[1]], 1, 0)])
    best = solver.tau[1][solver.op_to_idx[op[0]]]
    other = solver.tau[0][solver.op_to_idx[op[1]]]
    assert best == pytest.approx(3.0 * (solver.cfg.Q + 10.0)) and other == pytest.approx(solver.cfg.Q + 2.0)


def test_aco_hh_local_search_polishes_the_result() -> None:
    problem = _problem(15, 2)
    start = [[b for b in range(1, 16)]]  # one long trip in index order: room for local search
    plain = HyperHeuristicACO(problem, HyperHeuristicACOConfig(seed=1, max_iterations=1, n_ants=2), start).solve()
    polished = HyperHeuristicACO(
        problem,
        HyperHeuristicACOConfig(seed=1, max_iterations=1, n_ants=2, local_search=True, local_search_iterations=50),
        start,
    ).solve()
    assert polished[2] <= plain[2] + 1e-9  # never a longer plan
    mandatory = set(problem.mandatory_nodes)
    assert mandatory <= {b for r in polished[0] for b in r}


def test_hgs_mutation_rate_is_the_education_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offspring are educated by local search with probability mutation_rate (1.0: always, without a draw)."""
    for rate, expected in ((0.0, 0), (1.0, 20)):
        solver = HGSSolver(_problem(), HGSConfig(seed=1, mutation_rate=rate, mu=6, n_offspring=6))
        feasible, infeasible = solver._initialize_population(1.0)
        calls: List[int] = []
        real = solver.ls.optimize
        monkeypatch.setattr(solver.ls, "optimize", lambda ind, real=real, calls=calls: calls.append(1) or real(ind))
        for _ in range(20):
            solver._generate_offspring(feasible, infeasible, 1.0)
        assert len(calls) == expected


@pytest.mark.parametrize("profit_aware", [True, False])
def test_split_skips_unprofitable_bins_only_when_profit_aware(profit_aware: bool) -> None:
    rng = np.random.default_rng(0)
    d = np.sqrt(((rng.uniform(0, 10, (9, 2))[:, None] - rng.uniform(0, 10, (9, 2))[None]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    split = LinearSplit(d, {i: 10.0 for i in range(1, 9)}, 1000.0, 1e-4, 1.0, 0, [1])
    split.profit_aware = profit_aware
    routes, _ = split.split(list(range(1, 9)))
    kept = sorted(b for r in routes for b in r)
    assert kept == ([1] if profit_aware else list(range(1, 9)))


@pytest.mark.parametrize("profit_aware", [True, False])
def test_psoma_passes_profit_awareness_to_its_split(profit_aware: bool) -> None:
    psoma = PSOMASolver(_problem(revenue=1e-4), PSOMAConfig(profit_aware_operators=profit_aware))
    assert psoma.split_solver.profit_aware is profit_aware


@pytest.mark.parametrize(
    ("profit_aware", "expected"), [(True, "route_profit_gpx_crossover"), (False, "ordered_crossover")]
)
def test_hgs_switch_selects_the_crossover(monkeypatch: pytest.MonkeyPatch, profit_aware: bool, expected: str) -> None:
    from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search import hgs_population

    used: List[str] = []
    for name in ("route_profit_gpx_crossover", "ordered_crossover"):
        real = getattr(hgs_population, name)
        monkeypatch.setattr(
            hgs_population, name, lambda *a, real=real, name=name, **k: used.append(name) or real(*a, **k)
        )
    solver = HGSSolver(
        _problem(), HGSConfig(seed=1, profit_aware_operators=profit_aware, crossover_rate=1.0, mu=6, n_offspring=6)
    )
    feasible, infeasible = solver._initialize_population(1.0)
    for _ in range(5):
        solver._generate_offspring(feasible, infeasible, 1.0)
    assert used and set(used) == {expected}


def test_dssr_budget_and_switch_reach_column_generation() -> None:
    import inspect

    from logic.src.policies.helpers.solvers_and_matheuristics.search import bpc_tree, column_generation

    params = inspect.signature(column_generation.column_generation_loop).parameters
    assert params["enable_dssr"].default is True and params["dssr_max_iters"].default == 0
    source = inspect.getsource(bpc_tree)
    assert "enable_dssr=" in source and "dssr_max_iters=" in source and "enable_reduced_cost_arc_fixing" in source


def test_every_option_of_the_campaign_algorithms_is_read() -> None:
    """Static check over the modules each campaign algorithm imports: every option name is read somewhere."""
    import importlib
    import re
    import sys

    from logic.src.policies.routing.base.registry import RoutingAlgorithmRegistry as R

    for key in ("aco_hh", "pg_clns", "bpc", "alns", "swc_tcf", "hgs", "psoma", "sans", "cvrp"):
        cls = R.get(key)
        importlib.import_module(cls.__module__)
        text = "\n".join(
            Path(m.__file__).read_text()
            for name, m in list(sys.modules.items())
            if name.startswith("logic.src.") and getattr(m, "__file__", "") and "/configs/" not in m.__file__
        )
        for fld in dataclasses.fields(cls._config_class()):
            if fld.name in ("packing", "route_improvement"):
                continue
            assert re.search(rf"\.{fld.name}\b|['\"]{fld.name}['\"]", text), f"{key}: {fld.name} is never read"


def test_hgs_population_build_leaves_time_for_the_genetic_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local search slower than the limit allows used to fill the whole budget with the 4μ initial individuals, so
    no offspring was ever produced; the build now stops at init_time_share of the time left."""
    import time

    solver = HGSSolver(_problem(), HGSConfig(seed=1, time_limit=1.0, init_time_share=0.5, mu=25))
    real = solver.ls.optimize
    monkeypatch.setattr(solver.ls, "optimize", lambda ind: time.sleep(0.05) or real(ind))
    built: List[int] = []
    init = solver._initialize_population
    monkeypatch.setattr(
        solver,
        "_initialize_population",
        lambda pc, deadline=None: (lambda r: built.append(len(r[0]) + len(r[1])) or r)(init(pc, deadline)),
    )
    offspring: List[int] = []
    gen = solver._generate_offspring
    monkeypatch.setattr(solver, "_generate_offspring", lambda *a: offspring.append(1) or gen(*a))
    routes, _, _ = solver.solve()
    assert 2 <= built[0] < 4 * 25  # stopped growing at the deadline (100 builds would take 5 s)
    assert offspring  # and the genetic search ran
    assert HGSConfig().init_time_share == 0.5

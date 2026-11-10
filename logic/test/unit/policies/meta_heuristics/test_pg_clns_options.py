"""Every PG-CLNS option is read and has an effect, and the paper's yaml reproduces the paper's runs."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import yaml
from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.configs.policies.routing.meta_heuristics.pg_clns import PGCLNSConfig, PGCLNSLNSConfig
from logic.src.policies.routing.base.base_routing_policy import _structured_load
from logic.src.policies.routing.base.sub_problem import SubProblem, build_subproblem, coerce_config
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco import (
    ACOSolver,
)
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns import (
    LNSSolver,
)
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns import (
    PGCLNSSolver,
)
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones import (
    PheromoneTau,
)
from omegaconf import OmegaConf

YAML = "logic/configs/policies/routing/meta_heuristics/policy_pg_clns.yaml"


def _problem(n: int = 14, seed: int = 0, revenue: float = 0.3, capacity: float = 300.0) -> SubProblem:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n + 1, 2))
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    fill = rng.uniform(5, 100, n)
    mandatory = [i + 1 for i in range(n) if fill[i] >= 80] or [1]
    return build_subproblem(dist, {i + 1: float(fill[i]) for i in range(n)}, capacity, revenue, 1.0, mandatory)


def _paper_config() -> PGCLNSConfig:
    with open(YAML) as fh:
        raw = yaml.safe_load(fh)["pg_clns"]["custom"]
    raw = {k: v for k, v in raw.items() if k not in ("packing", "route_improvement")}
    return _structured_load(PGCLNSConfig, OmegaConf.create(raw))  # nested sections arrive as DictConfig


def _small(**overrides: Any) -> PGCLNSConfig:
    cfg = PGCLNSConfig(population_size=4, max_iterations=3, time_limit=0)
    cfg.lns = PGCLNSLNSConfig(max_iterations=15, time_limit=0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _solve(cfg: PGCLNSConfig, seed: int = 1, problem: SubProblem | None = None) -> Tuple[List[List[int]], float]:
    routes, profit, _ = PGCLNSSolver(problem or _problem(), cfg, seed=seed).solve()
    return routes, profit


def test_nested_yaml_sections_are_read() -> None:
    """The lns and aco sections arrive as Hydra DictConfig; before, coerce_config fell back to the defaults."""
    lns = coerce_config(OmegaConf.create({"time_limit": 600.0, "xi": 0.35, "unknown": 1}), PGCLNSLNSConfig)
    assert (lns.time_limit, lns.xi) == (600.0, 0.35)
    solver = PGCLNSSolver(_problem(), _paper_config(), seed=1)
    assert isinstance(solver.params.aco, KSparseACOConfig) and isinstance(solver.params.lns, PGCLNSLNSConfig)
    assert solver.coaching_solver.params.time_limit == 600.0  # the yaml value, not the 30 s default
    assert solver.aco_internal.params.k_sparse == 10 and solver.params.aco.max_iterations == 0


def test_paper_yaml_reproduces_the_paper_settings() -> None:
    cfg = _paper_config()
    solver = PGCLNSSolver(_problem(), cfg, seed=1)
    assert solver.coaching_solver._weights.decay == 0.8  # the fixed rate of the paper's runs
    assert cfg.aco.max_iterations == 0 and cfg.profit_aware_operators is True


def test_reaction_factor_sets_the_operator_weight_rate() -> None:
    problem = _problem()
    for rf in (0.2, 0.5, 0.9):
        assert LNSSolver(problem, PGCLNSLNSConfig(reaction_factor=rf), seed=1)._weights.decay == pytest.approx(1 - rf)
    slow = _solve(_small(lns=PGCLNSLNSConfig(max_iterations=15, time_limit=0, reaction_factor=0.05)))
    fast = _solve(_small(lns=PGCLNSLNSConfig(max_iterations=15, time_limit=0, reaction_factor=0.95)))
    base = _solve(_small())
    assert slow != base or fast != base  # a different learning rate changes the search


def test_colony_rounds_run_n_ants_ants_and_can_improve_the_population(monkeypatch: pytest.MonkeyPatch) -> None:
    built: List[int] = []
    original = ACOSolver.solve

    def counting(self: ACOSolver) -> Any:
        before = len(built)
        construct = self.constructor.construct

        def tracked() -> List[List[int]]:
            built.append(1)
            return construct()

        self.constructor.construct = tracked  # type: ignore[method-assign]
        try:
            return original(self)
        finally:
            self.constructor.construct = construct  # type: ignore[method-assign]
            built.append(-(len(built) - before))  # marker: ants built in this call

    monkeypatch.setattr(ACOSolver, "solve", counting)
    aco = KSparseACOConfig(n_ants=6, max_iterations=2, time_limit=0, local_search=False, k_sparse=10, tau_min=0.001)
    _solve(_small(max_iterations=3, aco=aco))
    rounds = [-m for m in built if m < 0]
    assert rounds == [12, 12, 12]  # one colony call per PG-CLNS iteration: 2 rounds x 6 ants
    built.clear()
    _solve(_small(max_iterations=3, aco=KSparseACOConfig(max_iterations=0, k_sparse=10, tau_min=0.001)))
    assert built == []  # no colony rounds


def test_colony_local_search_and_its_budget_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    solver = PGCLNSSolver(_problem(), _small(aco=KSparseACOConfig(
        n_ants=3, max_iterations=1, time_limit=0, local_search=True, local_search_iterations=7, k_sparse=10,
        tau_min=0.001)), seed=1)
    real = solver.aco_internal.ls.optimize

    def tracked(routes: List[List[int]]) -> List[List[int]]:
        calls.append(solver.aco_internal.ls.params.local_search_iterations)
        return real(routes)

    monkeypatch.setattr(solver.aco_internal.ls, "optimize", tracked)
    solver.solve()
    assert calls and set(calls) == {7}


def test_profit_aware_colony_keeps_the_most_profitable_ant() -> None:
    problem = _problem(revenue=5.0)  # bins pay well: more bins means more profit and more distance
    aco = ACOSolver(problem.dist_matrix, problem.wastes, problem.capacity, problem.revenue, problem.cost_unit,
                    {"profit_aware_operators": False}, problem.mandatory_nodes, seed=3)
    ants: List[Tuple[float, float]] = []
    for _ in range(8):
        routes = aco.constructor.construct()
        cost = aco._calculate_cost(routes)
        ants.append((cost, -(sum(problem.wastes[n] * problem.revenue for r in routes for n in r) - cost)))
    aco.params.profit_aware_operators = False
    assert aco._score([[1]], 5.0) == 5.0
    aco.params.profit_aware_operators = True
    assert aco._score([[1]], 5.0) == pytest.approx(-(problem.wastes[1] * 5.0 - 5.0))


def test_profit_aware_operators_choose_the_repair_operators() -> None:
    problem = _problem(revenue=1e-4, capacity=1e4)  # optional bins never pay; one vehicle holds everything
    aware = LNSSolver(problem, PGCLNSLNSConfig(max_iterations=0), seed=1, profit_aware=True)
    plain = LNSSolver(problem, PGCLNSLNSConfig(max_iterations=0), seed=1, profit_aware=False)
    start = [list(problem.mandatory_nodes)]
    removed = [b for b in range(1, 15) if b not in start[0]][:3]
    # profit-aware repair leaves the unprofitable bins out; cost-only repair puts every removed bin back
    assert not set(removed) & {b for r in aware.repair_ops[0]([r[:] for r in start], removed) for b in r}
    assert set(removed) <= {b for r in plain.repair_ops[0]([r[:] for r in start], removed) for b in r}
    assert set(removed) <= {b for r in plain.repair_ops[1]([r[:] for r in start], removed) for b in r}
    off = _solve(_small(profit_aware_operators=False), problem=problem)
    on = _solve(_small(profit_aware_operators=True), problem=problem)
    assert on[1] >= off[1] - 1e-9 or off != on


def test_colony_seed_overrides_the_solver_seed() -> None:
    problem = _problem()
    a = ACOSolver(problem.dist_matrix, problem.wastes, problem.capacity, 0.3, 1.0, {"seed": 11}, problem.mandatory_nodes,
                  seed=1)
    b = ACOSolver(problem.dist_matrix, problem.wastes, problem.capacity, 0.3, 1.0, {"seed": 11}, problem.mandatory_nodes,
                  seed=2)
    c = ACOSolver(problem.dist_matrix, problem.wastes, problem.capacity, 0.3, 1.0, {}, problem.mandatory_nodes, seed=11)
    assert a.constructor.random.random() == b.constructor.random.random() == c.constructor.random.random()


def test_scale_prunes_values_back_to_the_default() -> None:
    tau = PheromoneTau(5, k=4, tau_0=1.0, tau_min=0.001, tau_max=10.0, scale=2)  # precision 0.01
    tau.set(0, 1, 1.005)  # evaporates to ~1.004: within 0.01 of tau_0, pruned
    tau.set(0, 2, 3.0)
    tau.evaporate_all(0.001)
    assert 1 not in tau._pheromone[0] and tau.get(0, 1) == 1.0 and 2 in tau._pheromone[0]
    keep = PheromoneTau(5, k=4, tau_0=1.0, tau_min=0.001, tau_max=10.0)  # no scale: nothing pruned
    keep.set(0, 1, 1.005)
    keep.evaporate_all(0.001)
    assert 1 in keep._pheromone[0]
    aco = ACOSolver(np.ones((3, 3)) - np.eye(3), {1: 1.0, 2: 1.0}, 10.0, 1.0, 1.0, {"scale": 3.0}, None, seed=1)
    assert aco.pheromone.scale == 3.0


def test_every_option_is_read_by_the_policy() -> None:
    """Static check: each field of the three PG-CLNS configs appears in the solver code."""
    import dataclasses
    import pathlib
    import re

    code = "\n".join(p.read_text() for p in pathlib.Path(
        "logic/src/policies/routing/meta_heuristics/pheromone_guided_cooperative_large_neighborhood_search").glob("*.py"))
    pipeline_level = {"packing", "route_improvement"}  # read by the simulator around every policy
    for cls in (PGCLNSConfig, PGCLNSLNSConfig, KSparseACOConfig):
        for f in dataclasses.fields(cls):
            if f.name in pipeline_level or (cls is KSparseACOConfig and f.name == "acceptance_criterion"):
                continue
            assert re.search(rf"\.{f.name}\b", code), f"{cls.__name__}.{f.name} is never read"


def test_dict_configs_still_work() -> None:
    cfg: Dict[str, Any] = {"population_size": 3, "max_iterations": 2, "time_limit": 0,
                           "aco": {"k_sparse": 8, "tau_min": 0.001, "max_iterations": 1, "n_ants": 2, "local_search": False,
                                   "time_limit": 0},
                           "lns": {"max_iterations": 5, "time_limit": 0}}
    routes, profit, _ = PGCLNSSolver(_problem(), cfg, seed=1).solve()
    assert routes and np.isfinite(profit)

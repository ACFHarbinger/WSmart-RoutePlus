"""First-solve snapshots captured before the CR-14 acceptance field migration.

The a9d934816 engines used flat ALNS/PG schedules. These snapshots include
routes, objectives, temperature and the final solver RNG state on asymmetric instances.
"""

import hashlib
import json

import numpy as np
import pytest
from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.pg_clns import PGCLNSConfig
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns import (
    LNSSolver,
)

_EXPECTED = {
    (0, "alns"): "dea37f803be002abaee0d05694a304e742bb6623635d3be7bafca326b80a1668",
    (0, "pg"): "b1dbe1a6e6d6edb2650a282b60c3702ceb685727b355ae82cb8667dd90101a9b",
    (1, "alns"): "dada35e71d6ffa7021aca23191790925037957512755e29749d1b5e39eb2e440",
    (1, "pg"): "dd900b11ca498b83b41c7a5aadd9133a33cadbfa068b7a3d39655169985cfeb3",
    (2, "alns"): "b58bc6166b819e0f9dbe4ce6cd7e877e4ecfe043ad1d65f2d76ae7bced649b94",
    (2, "pg"): "4d6be622579deeedb897899cdb12e4839a601cef1eedfc426cb7e6c1fab6bbf2",
    (3, "alns"): "f9594feae7fea80bfd5a6fe42d86789f9bbdcf045b5cca1842a48774dd79aa67",
    (3, "pg"): "5e18d1bb1ddd15fbb35cb665df0fefe9c8f648fd29173865847f5b5138bd8807",
    (4, "alns"): "cca372a52c7d63fc6b60a3f3377838fe93c6a2fca320d6948977475db74f1282",
    (4, "pg"): "2ba0f89e55d2fb8fb9964f8cbc032237b534f67b214847d7b19016f2a2a7cd60",
    (5, "alns"): "93efa344bb446aad0f532d0740828f22c04d9e700e454c4e8db3fa7c942cc0e6",
    (5, "pg"): "a5706a382d5a56fcf5b179743c21f83f9bdfb6577af23ef86f3ed92a0e25fbfa",
    (6, "alns"): "c9fad9976ae7bac8be5f5780aef1c230c87fac9ef3d10f51e5c4a21fdc8547ef",
    (6, "pg"): "f0418707cfbbbf7ad39e4ca517d0d8c565958b34d7a9e2275547d5f592cd902a",
    (7, "alns"): "9a01c1ebe3b5f024e8b9aee3ca68c71e4ae6816b7164f6e72e3e53915a47934f",
    (7, "pg"): "3b5f843f15367e15034b03dc9e81c5f012771a849daeee6d9ad0b773eddcd311",
}


@pytest.mark.parametrize("seed,name", list(_EXPECTED))
def test_seeded_search_matches_pre_migration_snapshot(seed, name):
    rng = np.random.default_rng(seed)
    distances = rng.uniform(0.1, 10, (11, 11))
    np.fill_diagonal(distances, 0)
    problem = build_subproblem(distances, {i: float(i + 1) for i in range(1, 11)}, 30, 1.0, 0.5)
    if name == "alns":
        config = ALNSConfig(max_iterations=30, time_limit=0, seed=seed)
        engine = ALNSSolver
    else:
        config = PGCLNSConfig().lns
        config.max_iterations = 30
        config.time_limit = 0
        config.seed = seed
        engine = LNSSolver
    solver = engine(problem, config)
    routes, profit, cost = solver.solve()
    result = [seed, name, routes, profit, cost, solver.acceptance_criterion.get_state(), repr(solver.random.getstate())]
    assert hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest() == _EXPECTED[seed, name]

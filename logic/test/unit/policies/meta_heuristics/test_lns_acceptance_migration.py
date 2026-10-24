"""First-solve snapshots captured before the CR-14 acceptance field migration.

The a9d934816 engines used flat ALNS/PG schedules. These snapshots include
routes, objectives, temperature and the final solver RNG state on asymmetric instances.
The PG-CLNS ("pg") snapshots were re-pinned on 2026-10-24, when PG-CLNS moved to the shared
destroy and repair operators (owner ruling); the ALNS ones are unchanged.
"""

import hashlib
import json

import numpy as np
import pytest
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.pg_clns import PGCLNSConfig
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns import (
    LNSSolver,
)

_EXPECTED = {
    (0, "alns"): "dea37f803be002abaee0d05694a304e742bb6623635d3be7bafca326b80a1668",
    (0, "pg"): "01b68087f8c39bd0a3c730d62f5c80d38254288049bbf27bc5d06c25def3a7a4",
    (1, "alns"): "dada35e71d6ffa7021aca23191790925037957512755e29749d1b5e39eb2e440",
    (1, "pg"): "31ff98c36ad68417014b789c2ea163a75598b56546a48685ec42590e7fd6a113",
    (2, "alns"): "b58bc6166b819e0f9dbe4ce6cd7e877e4ecfe043ad1d65f2d76ae7bced649b94",
    (2, "pg"): "6d0203fc5834a5f3f2b8c8a2d69b22d57a3b11509c54efc677cb8fd90af8a272",
    (3, "alns"): "f9594feae7fea80bfd5a6fe42d86789f9bbdcf045b5cca1842a48774dd79aa67",
    (3, "pg"): "3319608e81b4ef3f75c7d0aeaacf0679befa3b9f413e993f557a73a95577bbe7",
    (4, "alns"): "cca372a52c7d63fc6b60a3f3377838fe93c6a2fca320d6948977475db74f1282",
    (4, "pg"): "56e9ad068d6ec7d0b79e400ecb8a6912442e3fb9b21008252e953246e37f5945",
    (5, "alns"): "93efa344bb446aad0f532d0740828f22c04d9e700e454c4e8db3fa7c942cc0e6",
    (5, "pg"): "0a9bd2d472fcb4a6aa97b7976e3062d540b427ec68f33eb268af6145c85eece9",
    (6, "alns"): "c9fad9976ae7bac8be5f5780aef1c230c87fac9ef3d10f51e5c4a21fdc8547ef",
    (6, "pg"): "6592d59d1bbbe202ee0bd8b0341b1e1112bf289f35563b80073811279c05e364",
    (7, "alns"): "9a01c1ebe3b5f024e8b9aee3ca68c71e4ae6816b7164f6e72e3e53915a47934f",
    (7, "pg"): "46d43b6f086be2294b8991a9caca1412fad6a864f8c73a0c92ade4f6582ec6dd",
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

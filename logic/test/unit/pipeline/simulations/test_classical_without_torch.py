"""Classical routing and simulator imports work without the neural extra."""

import subprocess
import sys
from pathlib import Path


def test_classical_simulator_and_solvers_without_neural_imports():
    root = Path(__file__).resolve().parents[5]
    script = r"""
import importlib.abc
import sys
class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensordict', 'torchrl', 'lightning', 'pytorch_lightning', 'wandb'}:
            raise ImportError('neural extra deliberately unavailable: ' + fullname)
sys.meta_path.insert(0, BlockNeural())
import numpy as np
import runpy
runpy.run_path("main.py", run_name="gate_classical_import")
from types import SimpleNamespace
from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
for name in ['alns', 'pg_clns', 'bpc', 'cvrp', 'hgs', 'psoma', 'aco_hh', 'sans', 'swc_tcf']:
    RoutingAlgorithmFactory.get_adapter(name)
from logic.src.pipeline.simulations.states.base.context import SimulationContext
from logic.src.pipeline.features.test.engine import run_simulator_test
from logic.src.pipeline.simulations.actions.fill import FillAction
from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.pg_clns import PGCLNSConfig
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns import LNSSolver
from logic.src.tracking.integrations.data import RuntimeDataTracker
problem = build_subproblem(np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]]), {1: 5., 2: 4.}, 20., 3., 1.)
for solver, config in [(ALNSSolver, ALNSConfig(max_iterations=4, time_limit=0, seed=7)),
                       (LNSSolver, PGCLNSConfig().lns)]:
    config.max_iterations = 4
    config.time_limit = 0
    routes, profit, cost = solver(problem, config).solve()
    assert routes and np.isfinite(profit) and np.isfinite(cost)
context = SimpleNamespace(bins=SimpleNamespace(load_filling=lambda day: (1, np.zeros(2), np.zeros(2), 0)), day=1, overflows=2)
FillAction().execute(context)
assert context.overflows == 3
tracker = RuntimeDataTracker(None)
fields = tracker._extract_fields({'stock': np.array([1., 3.])})
assert tracker._compute_stats(fields)['stock']['mean'] == 2.
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

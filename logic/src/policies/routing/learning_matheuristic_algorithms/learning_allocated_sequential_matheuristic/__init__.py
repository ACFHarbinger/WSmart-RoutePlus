r"""LBBD → ALNS → BPC → RL → SP pipeline package.

Attributes:
----------
LASMPipelinePolicy  Policy adapter (registers as "lasm").
LASMPipelineParams  Configuration dataclass.
RLController        Standalone LinUCB / offline PPO controller.
run_lasm_pipeline   Low-level orchestrator for drop-in use.
reset_rl_controller Force re-initialise the global RL state.

Example:
----------
>>> from <package>.lbbd_pipeline import run_lasm_pipeline, LASMPipelineParams
>>> p = LASMPipelineParams(alpha=0.5, time_limit=120, rl_mode="online")
>>> flat_route, profit, cost = run_lasm_pipeline(
...     bins, dist_matrix, env, values, binsids, mandatory,
...     n_vehicles=2, params=p,
... )

Policy registration (automatic on import)::

    from <package>.lbbd_pipeline import LASMPipelinePolicy
    policy = LASMPipelinePolicy()
"""

from .dispatcher import reset_rl_controller, run_lasm_pipeline
from .params import LASMPipelineParams
from .policy_lasm import LASMPolicy
from .rl_controller import RLController

__all__ = [
    "LASMPolicy",
    "LASMPipelineParams",
    "RLController",
    "run_lasm_pipeline",
    "reset_rl_controller",
]

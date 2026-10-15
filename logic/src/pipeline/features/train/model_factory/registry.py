"""registry.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Any, Dict

from logic.src.pipeline.rl import REINFORCE

from .constructive import _create_pomo, _create_symnco
from .hrl import _create_hrl
from .imitation import _create_adaptive_imitation, _create_imitation
from .ppo import _create_gdpo, _create_ppo_family


def _create_reinforce(cfg, policy, env, kw):
    """create reinforce.

    Args:
    cfg: Description of cfg.
    policy: Description of policy.
    env: Description of env.
    kw: Description of kw.

    Returns:
        Any: Description of return value.
    """
    return REINFORCE(**kw)


_ALGO_REGISTRY: Dict[str, Any] = {
    "ppo": lambda c, p, e, kw: _create_ppo_family("ppo", c, p, e, kw),
    "sapo": lambda c, p, e, kw: _create_ppo_family("sapo", c, p, e, kw),
    "gspo": lambda c, p, e, kw: _create_ppo_family("gspo", c, p, e, kw),
    "dr_grpo": lambda c, p, e, kw: _create_ppo_family("dr_grpo", c, p, e, kw),
    "gdpo": _create_gdpo,
    "pomo": _create_pomo,
    "symnco": _create_symnco,
    "hrl": _create_hrl,
    "imitation": _create_imitation,
    "adaptive_imitation": _create_adaptive_imitation,
    "reinforce": _create_reinforce,
}

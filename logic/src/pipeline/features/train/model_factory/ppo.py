"""ppo.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Any, Dict

import pytorch_lightning as pl

from logic.src.configs import Config
from logic.src.pipeline.learning import (
    DRGRPO,
    GDPO,
    GSPO,
    PPO,
    SAPO,
)

from .imitation import _create_critic_helper


def _create_ppo_family(algo_name: str, cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """create ppo family.

    Args:
    algo_name: Description of algo_name.
    cfg: Description of cfg.
    policy: Description of policy.
    env: Description of env.
    kw: Description of kw.

    Returns:
        Any: Description of return value.
    """
    critic = _create_critic_helper(policy, cfg)
    cls_map = {"ppo": PPO, "sapo": SAPO, "gspo": GSPO, "dr_grpo": DRGRPO}
    return cls_map[algo_name](critic=critic, **kw)


def _create_gdpo(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """create gdpo.

    Args:
    cfg: Description of cfg.
    policy: Description of policy.
    env: Description of env.
    kw: Description of kw.

    Returns:
        Any: Description of return value.
    """
    return GDPO(
        gdpo_objective_keys=cfg.rl.gdpo.objective_keys,
        gdpo_objective_weights=cfg.rl.gdpo.objective_weights,
        gdpo_conditional_key=cfg.rl.gdpo.conditional_key,
        gdpo_renormalize=cfg.rl.gdpo.renormalize,
        **kw,
    )

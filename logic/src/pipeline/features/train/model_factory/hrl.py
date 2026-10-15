"""hrl.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Any, Dict

import pytorch_lightning as pl

from logic.src.configs import Config
from logic.src.models.meta.hrl_manager import MandatoryManager
from logic.src.pipeline.rl import HRLModule


def _create_hrl(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """create hrl.

    Args:
    cfg: Description of cfg.
    policy: Description of policy.
    env: Description of env.
    kw: Description of kw.

    Returns:
        Any: Description of return value.
    """
    manager = MandatoryManager(device=cfg.device, hidden_dim=cfg.meta_rl.meta_hidden_dim)
    return HRLModule(manager=manager, worker=policy, env=env, lr=cfg.meta_rl.meta_lr)

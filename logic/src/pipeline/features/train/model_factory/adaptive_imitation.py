"""Adaptive imitation (IL to RL) model factory.

Attributes:
    _create_adaptive_imitation: Create adaptive imitation learning module

Example:
    >>> from logic.src.pipeline.features.train.model_factory.adaptive_imitation import _create_adaptive_imitation
    >>> _create_adaptive_imitation(cfg, policy, env, kw)
"""

from typing import Any, Dict

import pytorch_lightning as pl
from hydra.utils import instantiate

from logic.src.configs import Config
from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

from .imitation import _resolve_configs


def _create_adaptive_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """Create adaptive imitation learning module.

    Args:
        cfg: Full training configuration.
        policy: Student policy to train.
        env: Environment instance.
        kw: Additional keyword arguments for the module.

    Returns:
        Configured AdaptiveImitation module.
    """
    env_cfg, _, _ = _resolve_configs(cfg)

    # Validate policy config is provided
    if cfg.rl.adaptive_imitation.policy_config is None:
        raise ValueError("adaptive_imitation.policy_config must be provided for AdaptiveImitation")

    # Instantiate policy config if it has _target_ (Hydra config) or is a dict with _target_ key
    policy_config = cfg.rl.adaptive_imitation.policy_config
    if hasattr(policy_config, "_target_") or (isinstance(policy_config, dict) and "_target_" in policy_config):
        policy_config = instantiate(policy_config)

    il_keys = (
        "policy_config",
        "il_weight",
        "il_decay",
        "patience",
        "threshold",
        "decay_step",
        "epsilon",
        "loss_fn",
        "seed",
    )
    module_kw = {k: v for k, v in kw.items() if k not in il_keys}
    return AdaptiveImitation(
        policy_config=policy_config,
        env_name=getattr(env_cfg, "name", "ptp") if env_cfg else "ptp",
        il_weight=cfg.rl.adaptive_imitation.il_weight,
        il_decay=cfg.rl.adaptive_imitation.il_decay,
        patience=cfg.rl.adaptive_imitation.patience,
        threshold=cfg.rl.adaptive_imitation.threshold,
        decay_step=cfg.rl.adaptive_imitation.decay_step,
        epsilon=cfg.rl.adaptive_imitation.epsilon,
        loss_fn=cfg.rl.adaptive_imitation.loss_fn,
        seed=cfg.seed,
        **module_kw,
    )

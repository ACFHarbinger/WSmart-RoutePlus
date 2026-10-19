"""Imitation learning model factory.

Creates imitation learning modules from configuration.

Attributes:
    _create_imitation: Create imitation learning module
    _create_critic_helper: Create critic network from actor policy

Example:
    >>> from logic.src.pipeline.features.train.model_factory.imitation import _create_imitation
    >>> _create_imitation(cfg, policy, env, kw)
"""

from typing import Any, Dict

import pytorch_lightning as pl
from hydra.utils import instantiate

from logic.src.configs import Config
from logic.src.models.common.critic_network.policy import create_critic_from_actor


def _resolve_configs(cfg: Any):
    """Helper to dynamically resolve env, policy, and model based on the active task."""
    task = getattr(cfg, "task", "train")
    task_cfg = getattr(cfg, task, cfg)
    env_cfg = getattr(task_cfg, "env", getattr(cfg, "env", None))
    policy_cfg = getattr(task_cfg, "policy", task_cfg)
    model_cfg = getattr(policy_cfg, "model", getattr(cfg, "model", None))
    return env_cfg, policy_cfg, model_cfg


def _create_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """Create imitation learning module.

    Args:
        cfg: Full training configuration.
        policy: Student policy to train.
        env: Environment instance.
        kw: Additional keyword arguments for the module.

    Returns:
        Configured ImitationLearning module.
    """
    env_cfg, _, _ = _resolve_configs(cfg)

    # Validate policy config is provided
    if cfg.rl.imitation.policy_config is None:
        raise ValueError("imitation.policy_config must be provided for ImitationLearning")

    # Instantiate policy config if it has _target_ (Hydra config) or is a dict with _target_ key
    policy_config = cfg.rl.imitation.policy_config
    if hasattr(policy_config, "_target_") or (isinstance(policy_config, dict) and "_target_" in policy_config):
        policy_config = instantiate(policy_config)

    # Forward the full keyword set, as REINFORCE receives it: the base module reads the
    # training setup (cfg, train_time, batch sizes, curriculum) from it.
    module_kw = {k: v for k, v in kw.items() if k not in ("policy_config", "seed", "device", "loss_fn", "fill_from", "expert_cache", "precompute")}
    from logic.src.pipeline.learning.core.imitation import ImitationLearning

    return ImitationLearning(
        policy_config=policy_config,
        env_name=getattr(env_cfg, "name", "ptp") if env_cfg else "ptp",
        loss_fn=cfg.rl.imitation.loss_fn,
        seed=cfg.seed,
        device=cfg.device,
        fill_from=cfg.rl.imitation.fill_from,
        expert_cache=cfg.rl.imitation.expert_cache,
        precompute=cfg.rl.imitation.precompute,
        **module_kw,
    )


def _create_critic_helper(policy, cfg: Config) -> Any:
    """Create critic network from actor policy.

    Args:
        policy: Actor policy to create critic from.
        cfg: Training configuration.

    Returns:
        Initialized critic network.
    """
    env_cfg, _, model_cfg = _resolve_configs(cfg)
    enc = getattr(model_cfg, "encoder", None) if model_cfg else None

    return create_critic_from_actor(
        policy,
        env_name=getattr(env_cfg, "name", "ptp") if env_cfg else "ptp",
        embed_dim=getattr(enc, "embed_dim", 128) if enc else 128,
        hidden_dim=getattr(enc, "hidden_dim", 512) if enc else 512,
        n_layers=getattr(enc, "n_layers", 3) if enc else 3,
        n_heads=getattr(enc, "n_heads", 8) if enc else 8,
    )

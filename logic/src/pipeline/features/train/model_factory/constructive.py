"""constructive.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Any, Dict, cast

import pytorch_lightning as pl

from logic.src.configs import Config
from logic.src.pipeline.learning import (
    POMO,
    SymNCO,
)


def _create_pomo(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """create pomo.

    Args:
    cfg: Description of cfg.
    policy: Description of policy.
    env: Description of env.
    kw: Description of kw.

    Returns:
        Any: Description of return value.
    """
    explicit = {
        "num_augment": cfg.rl.pomo.num_augment,
        "augment_fn": cfg.rl.pomo.augment_fn,
        "num_starts": cfg.rl.pomo.num_starts,
        "mandatory_starts_only": cfg.rl.pomo.mandatory_starts_only,
    }
    for k in explicit:
        kw.pop(k, None)
    return POMO(
        num_augment=int(cast(Any, explicit["num_augment"])),
        augment_fn=explicit["augment_fn"],  # type: ignore
        num_starts=int(cast(Any, explicit["num_starts"])) if explicit["num_starts"] is not None else None,
        mandatory_starts_only=explicit["mandatory_starts_only"],  # type: ignore[arg-type]
        **kw,
    )


def _create_symnco(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """create symnco.

    Args:
    cfg: Description of cfg.
    policy: Description of policy.
    env: Description of env.
    kw: Description of kw.

    Returns:
        Any: Description of return value.
    """
    # Fallback to POMO config if SymNCO specific config is missing for some keys
    explicit = {
        "alpha": cfg.rl.symnco.alpha,
        "beta": cfg.rl.symnco.beta,
        "num_augment": (
            cfg.rl.symnco.num_augment
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "num_augment")
            else cfg.rl.pomo.num_augment
        ),
        "augment_fn": (
            cfg.rl.symnco.augment_fn
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "augment_fn")
            else cfg.rl.pomo.augment_fn
        ),
        "num_starts": (
            cfg.rl.symnco.num_starts
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "num_starts")
            else cfg.rl.pomo.num_starts
        ),
    }
    for k in explicit:
        kw.pop(k, None)
    return SymNCO(
        alpha=float(cast(Any, explicit["alpha"])) if explicit["alpha"] is not None else 1.0,
        beta=float(cast(Any, explicit["beta"])) if explicit["beta"] is not None else 1.0,
        num_augment=int(cast(Any, explicit["num_augment"])),
        augment_fn=explicit["augment_fn"],  # type: ignore
        num_starts=int(cast(Any, explicit["num_starts"])) if explicit["num_starts"] is not None else None,
        **kw,
    )

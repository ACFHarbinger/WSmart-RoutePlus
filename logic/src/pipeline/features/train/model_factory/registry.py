"""registry.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import registry
"""

import functools
from typing import Any, Callable, Dict, Optional

from logic.src.utils.registry import import_target

_FACTORY = "logic.src.pipeline.features.train.model_factory"
_PPO_FAMILY = ("ppo", "sapo", "gspo", "dr_grpo")

# Algorithm name -> "module:factory". Factories load on lookup, so the training stack imports
# only the algorithm a run uses (the release ships REINFORCE and imitation, Q-12).
_ALGO_REGISTRY: Dict[str, str] = {
    **{name: f"{_FACTORY}.ppo:_create_ppo_family" for name in _PPO_FAMILY},
    "gdpo": f"{_FACTORY}.ppo:_create_gdpo",
    "pomo": f"{_FACTORY}.constructive:_create_pomo",
    "symnco": f"{_FACTORY}.constructive:_create_symnco",
    "hrl": f"{_FACTORY}.hrl:_create_hrl",
    "imitation": f"{_FACTORY}.imitation:_create_imitation",
    "adaptive_imitation": f"{_FACTORY}.adaptive_imitation:_create_adaptive_imitation",
    "reinforce": f"{_FACTORY}.registry:_create_reinforce",
}


def _create_reinforce(cfg: Any, policy: Any, env: Any, kw: Dict[str, Any]) -> Any:
    """Create the REINFORCE module from the shared keyword set.

    Args:
        cfg: Full training configuration (unused).
        policy: Policy (already in ``kw``).
        env: Environment (already in ``kw``).
        kw: Module keyword arguments.

    Returns:
        The REINFORCE module.
    """
    from logic.src.pipeline.rl.core.reinforce import REINFORCE

    return REINFORCE(**kw)


def get_algorithm_factory(name: str) -> Optional[Callable[..., Any]]:
    """The factory ``(cfg, policy, env, kw) -> module`` of algorithm ``name``, or None."""
    target = _ALGO_REGISTRY.get(name)
    if target is None:
        return None
    factory = import_target(target)
    return functools.partial(factory, name) if name in _PPO_FAMILY else factory

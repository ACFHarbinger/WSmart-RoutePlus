"""
Reinforcement Learning subpackage for WSmart-Route.

This package contains PyTorch Lightning modules, algorithms, and utilities
for training neural routing policies using reinforcement learning.

Core Modules:
- core/: Base RL modules, loss functions, and baselines
- meta/: Meta-learning and multi-objective optimization
- hpo/: Hyperparameter optimization utilities
- features/: Training features and hooks

Registries:
- ``RL_ALGORITHM_REGISTRY``: Maps algorithm names to Lightning module classes.
- ``get_rl_algorithm(name)``: Factory function for looking up algorithm classes.

Attributes:
    RL_ALGORITHM_REGISTRY (dict): Maps algorithm names to Lightning module classes.
    get_rl_algorithm (function): Factory function for looking up algorithm classes.

Example:
    >>> from logic.src.pipeline.learning import get_rl_algorithm
    >>> algorithm_class = get_rl_algorithm("ppo")
    >>> algorithm = algorithm_class(env, model, loss_fn, optimizer, max_steps)
    >>> algorithm.train()
"""

from typing import Any

from logic.src.utils.registry import import_target, lazy_getattr

_CORE = "logic.src.pipeline.learning.core"
# Algorithm name -> "module:class". Classes load on lookup, so importing this package loads
# only the algorithm a run uses (the release ships REINFORCE and imitation, Q-12).
RL_ALGORITHM_REGISTRY = {
    "reinforce": f"{_CORE}.reinforce:REINFORCE",
    "ppo": f"{_CORE}.ppo:PPO",
    "a2c": f"{_CORE}.a2c:A2C",
    "sapo": f"{_CORE}.sapo:SAPO",
    "gspo": f"{_CORE}.gspo:GSPO",
    "gdpo": f"{_CORE}.gdpo:GDPO",
    "dr_grpo": f"{_CORE}.dr_grpo:DRGRPO",
    "pomo": f"{_CORE}.pomo:POMO",
    "symnco": f"{_CORE}.symnco:SymNCO",
    "imitation": f"{_CORE}.imitation:ImitationLearning",
    "adaptive_imitation": f"{_CORE}.adaptive_imitation:AdaptiveImitation",
    "hrl": "logic.src.pipeline.learning.meta.hrl:HRLModule",
    "meta_rl": "logic.src.pipeline.learning.meta.module:MetaRLModule",
    "mvmoe_pomo": f"{_CORE}.mvmoe_pomo:MVMoE_POMO",
    "mvmoe_am": f"{_CORE}.mvmoe_am:MVMoE_AM",
}
_LAZY = {target.rsplit(":", 1)[1]: target for target in RL_ALGORITHM_REGISTRY.values()}


def get_rl_algorithm(name: str) -> type:
    """
    Look up an RL algorithm class by its short name.

    Args:
        name: Algorithm name (e.g. "reinforce", "ppo", "pomo").

    Returns:
        The Lightning module class (not instantiated).

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if name not in RL_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown RL algorithm: {name!r}. Available: {sorted(RL_ALGORITHM_REGISTRY.keys())}")
    return import_target(RL_ALGORITHM_REGISTRY[name])


__all__ = [
    "REINFORCE",
    "PPO",
    "A2C",
    "SAPO",
    "GSPO",
    "DRGRPO",
    "GDPO",
    "HRLModule",
    "MetaRLModule",
    "POMO",
    "SymNCO",
    "ImitationLearning",
    "AdaptiveImitation",
    "MVMoE_POMO",
    "MVMoE_AM",
    "RL_ALGORITHM_REGISTRY",
    "get_rl_algorithm",
]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

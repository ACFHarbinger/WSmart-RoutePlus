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
    >>> from logic.src.pipeline.rl import get_rl_algorithm
    >>> algorithm_class = get_rl_algorithm("ppo")
    >>> algorithm = algorithm_class(env, model, loss_fn, optimizer, max_steps)
    >>> algorithm.train()
"""

# Names resolve lazily through rl.core (P-25): importing this package does
# not load every RL algorithm.
from typing import Any

_LAZY_CORE = [
    "A2C",
    "DRGRPO",
    "GDPO",
    "GSPO",
    "POMO",
    "PPO",
    "REINFORCE",
    "SAPO",
    "AdaptiveImitation",
    "HRLModule",
    "ImitationLearning",
    "MetaRLModule",
    "MVMoE_AM",
    "MVMoE_POMO",
    "SymNCO",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_CORE:
        from logic.src.pipeline.rl import core

        obj = getattr(core, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

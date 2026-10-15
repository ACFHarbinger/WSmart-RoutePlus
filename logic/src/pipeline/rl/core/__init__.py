"""
RL Pipeline module for WSmart-Route.

Attributes:
    None

Example:
    None
"""

# Algorithms load lazily (P-25 / Q-12a): importing REINFORCE or ImitationLearning
# must not pull every RL algorithm or the meta stack.
from typing import Any

from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.common.baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    MeanBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    SharedBaseline,
    WarmupBaseline,
    get_baseline,
)

_LAZY = {
    "A2C": "logic.src.pipeline.rl.core.a2c:A2C",
    "AdaptiveImitation": "logic.src.pipeline.rl.core.adaptive_imitation:AdaptiveImitation",
    "DRGRPO": "logic.src.pipeline.rl.core.dr_grpo:DRGRPO",
    "GDPO": "logic.src.pipeline.rl.core.gdpo:GDPO",
    "GSPO": "logic.src.pipeline.rl.core.gspo:GSPO",
    "ImitationLearning": "logic.src.pipeline.rl.core.imitation:ImitationLearning",
    "MVMoE_AM": "logic.src.pipeline.rl.core.mvmoe_am:MVMoE_AM",
    "MVMoE_POMO": "logic.src.pipeline.rl.core.mvmoe_pomo:MVMoE_POMO",
    "POMO": "logic.src.pipeline.rl.core.pomo:POMO",
    "PPO": "logic.src.pipeline.rl.core.ppo:PPO",
    "REINFORCE": "logic.src.pipeline.rl.core.reinforce:REINFORCE",
    "SAPO": "logic.src.pipeline.rl.core.sapo:SAPO",
    "SymNCO": "logic.src.pipeline.rl.core.symnco:SymNCO",
    "HRLModule": "logic.src.pipeline.rl.meta.hrl:HRLModule",
    "MetaRLModule": "logic.src.pipeline.rl.meta.module:MetaRLModule",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, _, attr = _LAZY[name].partition(":")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RL4COLitModule",
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "WarmupBaseline",
    "POMOBaseline",
    "MeanBaseline",
    "SharedBaseline",
    "get_baseline",
    "BASELINE_REGISTRY",
    "REINFORCE",
    "PPO",
    "A2C",
    "SAPO",
    "GSPO",
    "GDPO",
    "AdaptiveImitation",
    "DRGRPO",
    "ImitationLearning",
    "MetaRLModule",
    "HRLModule",
    "POMO",
    "SymNCO",
    "MVMoE_POMO",
    "MVMoE_AM",
]

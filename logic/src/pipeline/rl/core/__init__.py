"""
RL Pipeline module for WSmart-Route.

Attributes:
    None

Example:
    None
"""

from typing import Any

from logic.src.utils.registry import lazy_getattr

# Public names load on first access, so importing the package loads none of them.
_LAZY = {
    "RL4COLitModule": "logic.src.pipeline.rl.common.base:RL4COLitModule",
    "BASELINE_REGISTRY": "logic.src.pipeline.rl.common.baselines:BASELINE_REGISTRY",
    "Baseline": "logic.src.pipeline.rl.common.baselines:Baseline",
    "CriticBaseline": "logic.src.pipeline.rl.common.baselines:CriticBaseline",
    "ExponentialBaseline": "logic.src.pipeline.rl.common.baselines:ExponentialBaseline",
    "MeanBaseline": "logic.src.pipeline.rl.common.baselines:MeanBaseline",
    "NoBaseline": "logic.src.pipeline.rl.common.baselines:NoBaseline",
    "POMOBaseline": "logic.src.pipeline.rl.common.baselines:POMOBaseline",
    "RolloutBaseline": "logic.src.pipeline.rl.common.baselines:RolloutBaseline",
    "SharedBaseline": "logic.src.pipeline.rl.common.baselines:SharedBaseline",
    "WarmupBaseline": "logic.src.pipeline.rl.common.baselines:WarmupBaseline",
    "get_baseline": "logic.src.pipeline.rl.common.baselines:get_baseline",
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


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

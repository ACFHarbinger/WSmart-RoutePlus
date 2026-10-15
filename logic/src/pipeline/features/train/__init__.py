"""Training entry points.

Attributes:
    run_training: Run a training job.
    create_model: Build a model from its config.
    run_hpo: Run a hyper-parameter optimisation job (loaded lazily; the HPO
        stack is excluded from the 1.0.0 release package, owner ruling Q-6).

"""

from typing import Any

from hydra.core.config_store import ConfigStore

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.pipeline.rl.common.trainer import WSTrainer

from .engine import run_training
from .model_factory import create_model

# Register configuration for Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

__all__ = [
    "run_training",
    "run_hpo",
    "create_model",
    "get_env",
    "WSTrainer",
    "AttentionModelPolicy",
]


def __getattr__(name: str) -> Any:
    if name == "run_hpo":
        from .hpo import run_hpo

        return run_hpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

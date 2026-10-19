"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Any

from hydra.core.config_store import ConfigStore

from logic.src.configs import Config
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.pipeline.learning.common.trainer import WSTrainer
from logic.src.pipeline.learning.envs import get_env
from logic.src.utils.registry import lazy_getattr

from .engine import run_training
from .model_factory import create_model

# HPO is not released (Q-6); it loads only when called.
_LAZY = {"run_hpo": "logic.src.pipeline.features.train.hpo:run_hpo"}

# Register configuration for Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

__all__ = ["run_training", "run_hpo", "create_model", "get_env", "WSTrainer", "AttentionModelPolicy"]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

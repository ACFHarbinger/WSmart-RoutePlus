"""Common neural policy templates and base components.

This package provides abstract base classes and modular components for building
DRL policies, including support for autoregressive decoding, iterative
improvement, and transductive search.

Attributes:
    ConstructivePolicy: Base for models that build solutions step-by-step.
    AutoregressiveDecoder: Shared logic for sequential token generation.
    AutoregressiveEncoder: Neural feature extraction for routing states.
    ImprovementPolicy: Template for models that refine existing solutions.
    NonAutoregressivePolicy: Parallel output generation (e.g., heatmaps).
    TransductiveModel: Base for models with online optimization components.
    TimeTrackingPolicy: Timing wrapper for performance profiling.

Example:
    >>> from logic.src.models.common import ConstructivePolicy
    >>> class MyNewModel(ConstructivePolicy):
    ...     def pre_forward(self, td, env):
    ...         return self.encoder(td)
"""

from logic.src.models.common.critic_network.policy import CriticNetwork as CriticNetwork
from logic.src.models.common.critic_network.policy import create_critic_from_actor as create_critic_from_actor

from .autoregressive.constructive import ConstructivePolicy
from .autoregressive.decoder import AutoregressiveDecoder
from .autoregressive.encoder import AutoregressiveEncoder
from .autoregressive.policy import AutoregressivePolicy
from .critic_network.model import LegacyCriticNetwork as LegacyCriticNetwork
from .improvement.decoder import ImprovementDecoder
from .improvement.encoder import ImprovementEncoder
from .improvement.policy import ImprovementPolicy
from .non_autoregressive.decoder import NonAutoregressiveDecoder
from .non_autoregressive.encoder import NonAutoregressiveEncoder
from .non_autoregressive.policy import NonAutoregressivePolicy
from .time_tracking_policy import TimeTrackingPolicy
from .transductive.active_search import ActiveSearch
from .transductive.base import TransductiveModel
from .transductive.eas import EAS
from .transductive.eas_embeddings import EASEmb
from .transductive.eas_layers import EASLay

__all__ = [
    "AutoregressiveEncoder",
    "AutoregressiveDecoder",
    "AutoregressivePolicy",
    "ConstructivePolicy",
    "NonAutoregressiveEncoder",
    "NonAutoregressiveDecoder",
    "NonAutoregressivePolicy",
    "ImprovementEncoder",
    "ImprovementDecoder",
    "ImprovementPolicy",
    "TransductiveModel",
    "ActiveSearch",
    "EAS",
    "EASEmb",
    "EASLay",
    "CriticNetwork",
    "LegacyCriticNetwork",
    "create_critic_from_actor",
    "TimeTrackingPolicy",
]

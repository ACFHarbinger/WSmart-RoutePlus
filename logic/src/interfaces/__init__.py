"""
Interfaces for WSmart-Route logic components.
DEFINING PROTOCOLS TO DECOUPLE MODULES.

Attributes:
    IAcceptanceCriterion: Interface for acceptance criteria
    IBinContainer: Interface for bin containers
    IDistanceMetric: Interface for distance metrics
    IEnv: Interface for environments
    IPackingAlgorithm: Interface for packing algorithms
    IModel: Interface for models
    IPolicy: Interface for policies
    IRoutingAlgorithm: Interface for routing algorithms
    IRouteImprovement: Interface for route improvements
    ITensorDictLike: Interface for tensor dict like objects
    ITraversable: Interface for traversable objects

Example:
    >>> from logic.src.interfaces import (
        IAcceptanceCriterion,
        IBinContainer,
        IDistanceMetric,
        IEnv,
        IPackingAlgorithm,
        IModel,
        IPolicy,
        IRoutingAlgorithm,
        IRouteImprovement,
        ITensorDictLike,
        ITraversable,
    )
"""

from .acceptance_criterion import IAcceptanceCriterion
from .bin_container import IBinContainer
from .context import (
    AcceptanceMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)
from .env import IEnv
from .packing import IPackingAlgorithm
from .model import IModel
from .policy import IPolicy
from .routing_algorithm import IRoutingAlgorithm
from .route_improvement import IRouteImprovement
from .tensor_dict_like import ITensorDictLike
from .traversable import ITraversable

__all__ = [
    "IAcceptanceCriterion",
    "IRoutingAlgorithm",
    "IEnv",
    "IModel",
    "IPackingAlgorithm",
    "IPolicy",
    "IRouteImprovement",
    "ITensorDictLike",
    "ITraversable",
    "IBinContainer",
    # Context / tracking types
    "SearchContext",
    "SearchPhase",
    "SelectionMetrics",
    "ConstructionMetrics",
    "AcceptanceMetrics",
    "ImprovementMetrics",
    "merge_context",
]

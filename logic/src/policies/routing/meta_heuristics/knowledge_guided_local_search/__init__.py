"""
Knowledge-Guided Local Search (KGLS) matheuristic package.

KGLS integrates machine learning predictions (knowledge base) with local search
metaheuristics to bias search towards promising solution regions.

Attributes:
    KGLSConfig: Configuration parameters for KGLS.
    KGLSSolver: Core KGLS solver logic.
    KGLSPolicy: Policy adapter for KGLS.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.knowledge_guided_local_search import KGLSSolver
"""

from .kgls import KGLSSolver
from logic.src.configs.policies.kgls import KGLSConfig
from .policy_kgls import KGLSPolicy

__all__ = ["KGLSConfig", "KGLSSolver", "KGLSPolicy"]

"""
Knowledge-Guided Local Search (KGLS) matheuristic package.

KGLS integrates machine learning predictions (knowledge base) with local search
metaheuristics to bias search towards promising solution regions.

Attributes:
    KGLSConfig: Configuration parameters for KGLS.
    KGLSSolver: Core KGLS solver logic.
    KGLSPolicy: Policy adapter for KGLS.

"""

from logic.src.configs.policies.routing.meta_heuristics.kgls import KGLSConfig

from .kgls import KGLSSolver
from .policy_kgls import KGLSPolicy

__all__ = ["KGLSConfig", "KGLSSolver", "KGLSPolicy"]

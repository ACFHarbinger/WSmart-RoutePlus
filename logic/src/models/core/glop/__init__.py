"""GLOP: Global-Local Optimization Policy.

This package implements the GLOP architecture for large-scale routing. It
decomposes the problem into neural global partitioning and heuristic local
construction.

Attributes:
    GLOP: Unified training wrapper.
    GLOPPolicy: Hierarchical node partitioning policy.

"""

from .model import GLOP as GLOP
from .policy import GLOPPolicy as GLOPPolicy

__all__ = ["GLOP", "GLOPPolicy"]

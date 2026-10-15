"""DeepACO Neural Ant Colony Optimization components.

This package provides the implementation of DeepACO (Ye et al. 2023), which
couples GNN-based heatmap prediction with probabilistic ant construction.

Attributes:
    DeepACO: Primary training wrapper for neural ACO.
    DeepACOPolicy: The encoder-decoder policy for ant construction.

"""

from .model import DeepACO as DeepACO
from .policy import DeepACOPolicy as DeepACOPolicy

__all__ = ["DeepACO", "DeepACOPolicy"]

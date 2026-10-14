r"""K-Sparse Ant Colony Optimization.

This package implements the K-Sparse ACO algorithm, which restricts the
search space to the k-nearest neighbors for efficiency.

Attributes:
    KSparseACOSolver: The main solver class.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse import KSparseACOSolver
    >>> solver = KSparseACOSolver(dist_matrix, wastes, capacity, R, C, params)
"""

from .solver import KSparseACOSolver

__all__ = ["KSparseACOSolver"]

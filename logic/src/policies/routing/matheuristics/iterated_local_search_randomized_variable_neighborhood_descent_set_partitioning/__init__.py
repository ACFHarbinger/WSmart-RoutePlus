"""
ILS-RVND-SP matheuristic package.

Iterated Local Search with Randomized Variable Neighborhood Descent and
Set Partitioning post-optimization.

Attributes:
    ILSRVNDSPSolver: Core ILS-RVND-SP solver class.
    ILSRVNDSPParams: Configuration parameters dataclass.

Example:
    >>> from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning import ILSRVNDSPSolver, ILSRVNDSPParams
    >>> params = ILSRVNDSPParams()
    >>> solver = ILSRVNDSPSolver(dist_matrix, wastes, capacity, R, C, params)
"""

"""
Quantum-Inspired Differential Evolution (QDE) for VRPP.

Attributes:
    QDESolver: QDE solver mapping quantum amplitudes to discrete routing solutions.
    QDEConfig: Configuration parameters dataclass.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.quantum_differential_evolution import QDESolver, QDEConfig
    >>> params = QDEConfig(pop_size=20, max_iterations=200)
    >>> solver = QDESolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

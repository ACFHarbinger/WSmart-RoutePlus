"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for VRPP.

Attributes:
    PSOMASolver: Core PSOMA solver class.
    PSOMAParams: Configuration parameters dataclass.
    PSOMAParticle: Particle representation for the swarm.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_memetic_algorithm import PSOMASolver, PSOMAParams
    >>> params = PSOMAParams(pop_size=20, max_iterations=200)
    >>> solver = PSOMASolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

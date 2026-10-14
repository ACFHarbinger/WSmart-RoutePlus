"""
Sequence-based Selection Hyper-Heuristic (SS-HH) for VRPP.

This module implements the Sequence-based Selection Hyper-Heuristic (SS-HH)
solver for the Vehicle Routing Problem with Profit (VRPP). The SS-HH is an
online-learning hyper-heuristic that models the search as a sequence of moves.
It learns the transition probabilities between low-level heuristics based on their
historical performance in discovering improving or intensifying solutions.

Bibliography:
    - Kheiri, A. "Heuristic Sequence Selection for Inventory Routing Problem", 2014.
    - Bibliography: bibliography/Sequence-based_Selection_Hyper-Heuristic.pdf

Attributes:
    SSHHPolicy: Adapter for the SS-HH solver.
    SSHHParams: Configuration parameters for the SS-HH solver.
    SSHHSolver: Implements the Sequence-based Selection Hyper-Heuristic.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics import SSHHPolicy
    >>> params = SSHHPolicy()
    >>> print(params)
    SSHHParams(
        max_iterations=500,
        n_removal=2,
        n_llh=5,
        time_limit=60.0,
        threshold_infeasible=0.001,
        threshold_feasible_base=0.0001,
        threshold_decay_rate=0.01,
        vrpp=True,
        profit_aware_operators=False,
        seed=None,
    )
"""

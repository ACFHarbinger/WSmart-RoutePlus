"""Configuration parameters for the Guided Local Search (GLS) solver.

Attributes:
    GLSParams: Parameter dataclass for the Guided Local Search.

Example:
    >>> params = GLSParams(lambda_param=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GLSParams:
    """Configuration for the GLS solver.

    Attributes:
        lambda_param: Scaling coefficient for the penalty term (global intensity).
        alpha_param: Tuning parameter for the static base_lambda calculation.
        penalty_cycles: Number of GLS penalty update cycles (restarts).
        n_removal: Maximum nodes removed per LNS ruin step.
        n_llh: Number of Low-Level Heuristics in the pool.
        inner_iterations: Stagnation threshold for declaring a local optimum.
        fls_coupling_prob: Probability of triggering targeted penalized removal.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
    """

    lambda_param: float = 1.0
    alpha_param: float = 0.3
    penalty_cycles: int = 1000
    n_removal: int = 2
    n_llh: int = 6
    inner_iterations: int = 100
    fls_coupling_prob: float = 0.8
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

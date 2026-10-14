"""Configuration for the Logic-Based Benders Decomposition (LBBD) policy.

Attributes:
    LBBDConfig: Configuration for the Logic-Based Benders Decomposition (LBBD) policy.

Example:
    >>> from configs.policies.lbbd import LBBDConfig
    >>> config = LBBDConfig()
    >>> config.time_limit
    300.0
    >>> config.k
    10
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LBBDConfig:
    """
    Configuration for the Logic-Based Benders Decomposition (LBBD) policy.

    Attributes:
        num_days (int): Number of days for multi-day horizon.
        stochastic_master (bool): Whether to model multiple scenarios explicitly in the Master Problem.
        mean_increment (float): Mean fractional increment per day (0.0 to 1.0).
        num_scenarios (int): Number of scenarios for SAA in Subproblem routing cost estimation (if used).
        max_iterations (int): Maximum number of iterations.
        benders_gap (float): Benders gap tolerance.
        time_limit (float): Time limit in seconds.
        subproblem_timeout (float): Time limit for subproblem.
        mip_gap (float): MIP gap tolerance.
        waste_weight (float): Weight for waste.
        cost_weight (float): Weight for cost.
        overflow_penalty (float): Penalty for overflow.
        use_nogood_cuts (bool): Whether to use Nogood cuts for feasibility.
        use_optimality_cuts (bool): Whether to use logic-based optimality cuts for travel cost.
        seed (Optional[int]): Seed for random number generator.
        vrpp (bool): Whether the problem is a VRRP.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (Optional[List[Any]]): Mandatory customers/requests selection.
        route_improvement (Optional[List[Any]]): Route improvement strategies.
    """

    # Multi-day horizon
    num_days: int = 3
    # Whether to model multiple scenarios explicitly in the Master Problem
    stochastic_master: bool = False

    # Mean fractional increment per day (0.0 to 1.0)
    mean_increment: float = 0.2

    # Number of scenarios for SAA in Subproblem routing cost estimation (if used)
    # Note: LBBD typically assumes deterministic subproblems or expected costs.
    num_scenarios: int = 5

    # Algorithm parameters
    max_iterations: int = 20
    benders_gap: float = 0.01

    # Solve time limits
    time_limit: float = 300.0
    subproblem_timeout: float = 20.0
    mip_gap: float = 0.05

    # Objective weights
    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0

    # Whether to use Nogood cuts for feasibility
    use_nogood_cuts: bool = True
    # Whether to use logic-based optimality cuts for travel cost
    use_optimality_cuts: bool = True

    # Seed for reproducibility
    seed: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_days < 1:
            raise ValueError("num_days must be at least 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

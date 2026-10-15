"""
BC (Branch-and-Cut) configuration.

Attributes:
    BCConfig: Attributes for BC configuration.

Example:
    >>> from configs.policies.bc import BCConfig
    >>> config = BCConfig()
    >>> config.time_limit
    300.0
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class BCConfig:
    """Configuration for Branch-and-Cut (BC) policy.

    Branch-and-Cut is an exact optimization method that combines:
    - Cutting planes to strengthen LP relaxations
    - Branch-and-bound for integer optimization
    - Gurobi solver for MIP solving

    The algorithm solves the VRPP to provable optimality using:
    1. Initial LP relaxation
    2. Separation algorithms for violated inequalities:
       - Subtour elimination constraints (SEC)
       - Capacity inequalities
       - Comb inequalities (optional)
    3. Branch-and-bound when cuts are insufficient

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        mip_gap: Relative MIP optimality gap (0.0 = prove optimality).
        use_heuristics: Whether to use primal heuristics for warm start.
        max_cuts_per_round: Maximum cuts to add per separation round.
        enable_fractional_capacity_cuts: Enable exact fractional RCC separation.
            True = Use O(V⁴) max-flow for fractional capacity cuts (small instances).
            False = Disable exact fractional capacity separation (large instances).
            Recommended: True for n ≤ 75, False for n > 75.
            Note: Automatically disabled for instances with n > 75 regardless of setting.
        profit_aware_operators: Whether to use profit-aware operators.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
        use_saa: Whether to use Sample Average Approximation (SAA) for SIRP.
        num_scenarios: Number of scenarios to generate for SAA.
    """

    time_limit: float = 300.0
    mip_gap: float = 0.0
    use_heuristics: bool = True
    max_cuts_per_round: int = 50
    enable_fractional_capacity_cuts: bool = True
    profit_aware_operators: bool = False
    use_saa: bool = False
    num_scenarios: int = 10
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
    verbose: bool = False
    enable_heuristic_rcc_separation: bool = True
    enable_exact_rcc_separation: bool = False
    use_comb_cuts: bool = False


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BCConfig":
        """Create a BCConfig instance from a configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A BCConfig instance with values mapped from the config.
        """
        return cls(
            time_limit=config.get("time_limit", 300.0),
            mip_gap=config.get("mip_gap", 0.01),
            max_cuts_per_round=config.get("max_cuts_per_round", 50),
            use_heuristics=config.get("use_heuristics", True),
            verbose=config.get("verbose", False),
            profit_aware_operators=config.get("profit_aware_operators", False),
            enable_fractional_capacity_cuts=config.get("enable_fractional_capacity_cuts", True),
            enable_heuristic_rcc_separation=config.get("enable_heuristic_rcc_separation", True),
            enable_exact_rcc_separation=config.get("enable_exact_rcc_separation", False),
            use_comb_cuts=config.get("use_comb_cuts", False),
            use_saa=config.get("use_saa", False),
            num_scenarios=config.get("num_scenarios", 10),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter object to a serializable dictionary.

        Returns:
            A dictionary containing all parameter values.
        """
        return {
            "time_limit": self.time_limit,
            "mip_gap": self.mip_gap,
            "max_cuts_per_round": self.max_cuts_per_round,
            "use_heuristics": self.use_heuristics,
            "verbose": self.verbose,
            "profit_aware_operators": self.profit_aware_operators,
            "enable_fractional_capacity_cuts": self.enable_fractional_capacity_cuts,
            "enable_heuristic_rcc_separation": self.enable_heuristic_rcc_separation,
            "enable_exact_rcc_separation": self.enable_exact_rcc_separation,
            "use_comb_cuts": self.use_comb_cuts,
            "use_saa": self.use_saa,
            "num_scenarios": self.num_scenarios,
        }

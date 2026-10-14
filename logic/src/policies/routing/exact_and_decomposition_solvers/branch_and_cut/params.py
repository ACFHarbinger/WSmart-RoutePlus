"""Configuration parameters for the Branch-and-Cut (BC) algorithm.

Attributes:
    BCParams (class): Data structure for BC solver parameters.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.params import BCParams
    >>> params = BCParams(time_limit=300.0)
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BCParams:
    """Standardized parameters for the Branch-and-Cut solver.

    Attributes:
        time_limit (float): Maximum runtime in seconds. Defaults to 300.0.
        mip_gap (float): Relative optimality gap threshold. Defaults to 0.01.
        max_cuts_per_round (int): Maximum cuts per round. Defaults to 50.
        use_heuristics (bool): Enable heuristic warmstarts. Defaults to True.
        verbose (bool): Enable detailed logging. Defaults to False.
        profit_aware_operators (bool): Use profit-centric heuristics. Defaults to False.
        vrpp (bool): Problem type flag for VRPP. Defaults to False.
        enable_fractional_capacity_cuts (bool): Enable fractional capacity cuts.
        enable_heuristic_rcc_separation (bool): Enable heuristic RCC separation.
        enable_exact_rcc_separation (bool): Enable exact RCC separation.
        use_comb_cuts (bool): Enable comb cuts. Defaults to False.
        use_saa (bool): Enable Sample Average Approximation. Defaults to False.
        num_scenarios (int): Number of SAA scenarios. Defaults to 10.
    """

    time_limit: float = 300.0
    mip_gap: float = 0.01
    max_cuts_per_round: int = 50
    use_heuristics: bool = True
    verbose: bool = False
    profit_aware_operators: bool = False
    vrpp: bool = False
    enable_fractional_capacity_cuts: bool = True
    enable_heuristic_rcc_separation: bool = True
    enable_exact_rcc_separation: bool = False
    use_comb_cuts: bool = False
    use_saa: bool = False
    num_scenarios: int = 10

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BCParams":
        """Create a BCParams instance from a configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A BCParams instance with values mapped from the config.
        """
        return cls(
            time_limit=config.get("time_limit", 300.0),
            mip_gap=config.get("mip_gap", 0.01),
            max_cuts_per_round=config.get("max_cuts_per_round", 50),
            use_heuristics=config.get("use_heuristics", True),
            verbose=config.get("verbose", False),
            profit_aware_operators=config.get("profit_aware_operators", False),
            vrpp=config.get("vrpp", False),
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
            "vrpp": self.vrpp,
            "enable_fractional_capacity_cuts": self.enable_fractional_capacity_cuts,
            "enable_heuristic_rcc_separation": self.enable_heuristic_rcc_separation,
            "enable_exact_rcc_separation": self.enable_exact_rcc_separation,
            "use_comb_cuts": self.use_comb_cuts,
            "use_saa": self.use_saa,
            "num_scenarios": self.num_scenarios,
        }

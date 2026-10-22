"""
ILS-RVND-SP (Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning) configuration dataclasses.
Attributes:
    ILSRVNDSPConfig: Configuration for the ILS-RVND-SP algorithm.

Example:
    >>> from configs.policies.ils_rvnd_sp import ILSRVNDSPConfig
    >>> config = ILSRVNDSPConfig()
    >>> config.time_limit
    120.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig

from logic.src.configs.policies.routing.meta_heuristics.abc import ABCConfig


@dataclass
class ILSRVNDSPConfig(ABCConfig):
    """Configuration for the ILS-RVND-SP algorithm.

    Attributes:
        max_restarts (int): Maximum number of restarts for ILS.
        max_iter_ils (int): Maximum number of iterations for ILS.
        perturbation_strength (int): Strength of perturbation.
        use_set_partitioning (bool): Whether to use Set Partitioning.
        mip_time_limit (float): Time limit for MIP solver.
        sp_mip_gap (float): Gap tolerance for MIP solver.
        N (int): Parameter N.
        A (float): Parameter A.
        MaxIter_a (int): Maximum iterations for phase a.
        MaxIter_b (int): Maximum iterations for phase b.
        MaxIterILS_b (int): Maximum iterations for ILS in phase b.
        TDev_a (float): Temperature deviation for phase a.
        TDev_b (float): Temperature deviation for phase b.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        time_limit (float): Time limit for the algorithm.
        seed (Optional[int]): Seed for the random number generator.
        engine (str): Solver engine to use.
        framework (str): Framework for the algorithm.
        local_search_iterations (int): Number of local search iterations.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
        acceptance_criterion (AcceptanceConfig): Acceptance criterion configuration.
    """

    # ILS Phase parameters
    max_restarts: int = 10
    max_iter_ils: int = 50
    perturbation_strength: int = 2

    # Set Partitioning (SP) Phase parameters
    use_set_partitioning: bool = True
    mip_time_limit: float = 60.0
    sp_mip_gap: float = 0.01

    # Paper-specific advanced parameters
    N: int = 150
    A: float = 11.0
    MaxIter_a: int = 50
    MaxIter_b: int = 100
    MaxIterILS_b: int = 2000
    TDev_a: float = 0.05
    TDev_b: float = 0.005

    # Global parameters
    profit_aware_operators: bool = False
    time_limit: float = 120.0
    seed: Optional[int] = None
    engine: str = "gurobi"
    framework: str = "ortools"
    local_search_iterations: int = 500

    # --- Infrastructure Hooks ---
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="only_improving"))
    max_vehicles: int = 0


    def __post_init__(self):
        """Ensure acceptance criterion is initialized even if not passed in config."""
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
        if self.acceptance_criterion is None:
            # Standard ILS-RVND-SP uses Improving-Only acceptance
            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

    @classmethod
    def from_config(cls, config: Any) -> "ILSRVNDSPConfig":
        """Build parameters from a configuration object.

        Args:
            config: Configuration object with ILS-RVND-SP parameter attributes.

        Returns:
            ILSRVNDSPConfig: Populated parameter dataclass with acceptance criterion.
        """
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
        # Build parameters
        params = cls(
            max_restarts=getattr(config, "max_restarts", 10),
            max_iter_ils=getattr(config, "max_iter_ils", 100),
            perturbation_strength=getattr(config, "perturbation_strength", 2),
            use_set_partitioning=getattr(config, "use_set_partitioning", True),
            mip_time_limit=getattr(config, "mip_time_limit", 60.0),
            sp_mip_gap=getattr(config, "sp_mip_gap", 0.01),
            N=getattr(config, "N", 150),
            A=getattr(config, "A", 11.0),
            MaxIter_a=getattr(config, "MaxIter_a", 50),
            MaxIter_b=getattr(config, "MaxIter_b", 100),
            MaxIterILS_b=getattr(config, "MaxIterILS_b", 2000),
            TDev_a=getattr(config, "TDev_a", 0.05),
            TDev_b=getattr(config, "TDev_b", 0.005),
            max_vehicles=getattr(config, "max_vehicles", 9999),
            time_limit=getattr(config, "time_limit", 300.0),
            seed=getattr(config, "seed", 42),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
        )

        # Handle Acceptance Criterion Injection

        acceptance_cfg = getattr(config, "acceptance_criterion", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to only_improving for standard ILS-RVND-SP
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

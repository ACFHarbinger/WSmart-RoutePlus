"""
ILS-RVND-SP algorithm parameters.

Attributes:
    ILSRVNDSPParams: Dataclass with ILS-RVND-SP runtime configuration.

Example:
    >>> from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.params import ILSRVNDSPParams
    >>> params = ILSRVNDSPParams(max_restarts=10, time_limit=300.0)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory


@dataclass
class ILSRVNDSPParams:
    """
    Runtime parameters for the ILS-RVND-SP solver.

    Attributes:
        max_restarts: Number of independent restarts from different initial solutions.
        max_iter_ils: Maximum ILS iterations per restart.
        perturbation_strength: Number of nodes removed during perturbation.
        use_set_partitioning: Whether to apply set partitioning post-optimization.
        mip_time_limit: Time budget for set partitioning MIP.
        sp_mip_gap: Relative gap tolerance for set partitioning.
        N: Maximum number of nodes for subgradient optimization.
        time_limit: Total wall-clock time budget.
        seed: Random seed.
        vrpp: Whether to operate in VRPP profit-aware mode.
    """

    max_restarts: int = 10
    max_iter_ils: int = 100
    perturbation_strength: int = 2

    use_set_partitioning: bool = True
    mip_time_limit: float = 60.0
    sp_mip_gap: float = 0.01

    N: int = 150
    A: float = 11.0
    MaxIter_a: int = 50
    MaxIter_b: int = 100
    MaxIterILS_b: int = 2000
    TDev_a: float = 0.05
    TDev_b: float = 0.005
    max_vehicles: int = 0

    time_limit: float = 300.0
    seed: Optional[int] = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    local_search_iterations: int = 500

    engine: str = "gurobi"
    framework: str = "ortools"

    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    def __post_init__(self):
        """Ensure acceptance criterion is initialized even if not passed in config."""
        if self.acceptance_criterion is None:
            # Standard ILS-RVND-SP uses Improving-Only acceptance
            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

    @classmethod
    def from_config(cls, config: Any) -> "ILSRVNDSPParams":
        """Build parameters from a configuration object.

        Args:
            config: Configuration object with ILS-RVND-SP parameter attributes.

        Returns:
            ILSRVNDSPParams: Populated parameter dataclass with acceptance criterion.
        """
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
            vrpp=getattr(config, "vrpp", True),
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

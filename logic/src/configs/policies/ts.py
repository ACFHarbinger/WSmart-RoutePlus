"""
TS (Tabu Search) configuration for Hydra.

Attributes:
    TSConfig: Configuration for the Tabu Search (TS) policy.

Example:
    >>> from configs.policies.ts import TSConfig
    >>> config = TSConfig()
    >>> config.max_iterations
    5000
    >>> config.time_limit
    60.0
    >>> config.packing
    []
    >>> config.route_improvement
    []
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig


@dataclass
class TSConfig:
    """Configuration for the Tabu Search policy.

    Attributes:
        tabu_tenure: Tenure for the short-term memory (number of iterations to keep moves tabu).
        dynamic_tenure: Whether to dynamically adjust tabu tenure during search.
        min_tenure: Minimum tabu tenure value when dynamic tenure is enabled.
        max_tenure: Maximum tabu tenure value when dynamic tenure is enabled.
        aspiration_enabled: Whether to allow moves that violate tabu status if they improve the solution.
        intensification_enabled: Whether to use intensification (focusing on high-performing regions).
        diversification_enabled: Whether to use diversification (exploring new regions).
        intensification_interval: Number of iterations between intensification phases.
        diversification_interval: Number of iterations between diversification phases.
        elite_size: Number of elite solutions to maintain for intensification.
        frequency_penalty_weight: Weight for frequency-based penalties in long-term memory.
        candidate_list_enabled: Whether to use candidate lists to limit neighborhood exploration.
        candidate_list_size: Maximum number of candidates to consider from the neighborhood.
        oscillation_enabled: Whether to use strategic oscillation to escape local optima.
        feasibility_tolerance: Tolerance for infeasibility when using strategic oscillation.
        max_iterations: Maximum number of iterations to run the search.
        max_iterations_no_improve: Maximum number of iterations without improvement before stopping.
        n_removal: Number of routes/sequences to remove in each iteration (for destroying and rebuilding).
        n_llh: Number of local search operators to apply in each iteration.
        time_limit: Maximum time in seconds to run the search.
        use_swap: Whether to include swap moves in the neighborhood.
        use_relocate: Whether to include relocate moves in the neighborhood.
        use_2opt: Whether to include 2-opt moves in the neighborhood.
        use_insertion: Whether to include insertion moves in the neighborhood.
        seed: Random seed for reproducibility.
        profit_aware_operators: Whether to use profit-aware considerations in neighborhood operators.
        packing: List of packing algorithms to apply.
        route_improvement: List of route improvement strategies to apply.
        acceptance_criterion: Configuration for the acceptance criterion (e.g., simulated annealing).
    """

    # Short-term memory (Recency-based)
    tabu_tenure: int = 7
    dynamic_tenure: bool = True
    min_tenure: int = 5
    max_tenure: int = 15

    # Aspiration criteria
    aspiration_enabled: bool = True

    # Long-term memory (Frequency-based)
    intensification_enabled: bool = True
    diversification_enabled: bool = True
    intensification_interval: int = 100
    diversification_interval: int = 200
    elite_size: int = 5
    frequency_penalty_weight: float = 0.1

    # Candidate list strategies
    candidate_list_enabled: bool = True
    candidate_list_size: int = 20

    # Strategic oscillation
    oscillation_enabled: bool = False
    feasibility_tolerance: float = 0.1

    # General search parameters
    max_iterations: int = 5000
    max_iterations_no_improve: int = 500
    n_removal: int = 3
    n_llh: int = 5
    time_limit: float = 60.0

    # Neighborhood structure
    use_swap: bool = True
    use_relocate: bool = True
    use_2opt: bool = True
    use_insertion: bool = True

    # Standard fields
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="ac"))


    def __post_init__(self):
        """Ensure acceptance criterion is initialized.

        Returns:
            None.
        """
        if self.acceptance_criterion is None:
            # Standard Tabu Search uses Aspiration Criterion acceptance
            from logic.src.policies.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="ac")

    @classmethod
    def from_config(cls, config: Any) -> "TSConfig":
        """Build parameters from a configuration source.

        Args:
            config: Configuration source.

        Returns:
            Instantiated TSConfig.
        """
        # Build parameters
        params = cls(
            tabu_tenure=getattr(config, "tabu_tenure", 7),
            dynamic_tenure=getattr(config, "dynamic_tenure", True),
            min_tenure=getattr(config, "min_tenure", 5),
            max_tenure=getattr(config, "max_tenure", 15),
            aspiration_enabled=getattr(config, "aspiration_enabled", True),
            intensification_enabled=getattr(config, "intensification_enabled", True),
            diversification_enabled=getattr(config, "diversification_enabled", True),
            intensification_interval=getattr(config, "intensification_interval", 100),
            diversification_interval=getattr(config, "diversification_interval", 200),
            elite_size=getattr(config, "elite_size", 5),
            frequency_penalty_weight=getattr(config, "frequency_penalty_weight", 0.1),
            candidate_list_enabled=getattr(config, "candidate_list_enabled", True),
            candidate_list_size=getattr(config, "candidate_list_size", 20),
            oscillation_enabled=getattr(config, "oscillation_enabled", False),
            feasibility_tolerance=getattr(config, "feasibility_tolerance", 0.1),
            max_iterations=getattr(config, "max_iterations", 5000),
            max_iterations_no_improve=getattr(config, "max_iterations_no_improve", 500),
            n_removal=getattr(config, "n_removal", 3),
            n_llh=getattr(config, "n_llh", 5),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            use_swap=getattr(config, "use_swap", True),
            use_relocate=getattr(config, "use_relocate", True),
            use_2opt=getattr(config, "use_2opt", True),
            use_insertion=getattr(config, "use_insertion", True),
        )

        # Handle Acceptance Criterion Injection
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = getattr(config, "acceptance_criterion", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to aspiration for standard TS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="ac")

        return params

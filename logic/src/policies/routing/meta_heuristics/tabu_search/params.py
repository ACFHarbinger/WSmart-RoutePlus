"""Configuration parameters for the Tabu Search (TS) solver.

Attributes:
    TSParams: Parameter dataclass for the Tabu Search.

Example:
    >>> params = TSParams(max_iterations=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class TSParams:
    """Configuration for the Tabu Search solver.

    Attributes:
        tabu_tenure: Fixed tabu tenure.
        dynamic_tenure: Whether to use dynamic tenure.
        min_tenure: Minimum tenure for dynamic adjustment.
        max_tenure: Maximum tenure for dynamic adjustment.
        aspiration_enabled: Enable aspiration criteria.
        intensification_enabled: Enable intensification strategies.
        diversification_enabled: Enable diversification strategies.
        intensification_interval: Iterations between intensification phases.
        diversification_interval: Iterations between diversification phases.
        elite_size: Number of elite solutions to maintain.
        frequency_penalty_weight: Weight for frequency-based penalties.
        candidate_list_enabled: Use candidate list to restrict neighborhood.
        candidate_list_size: Maximum size of candidate list.
        oscillation_enabled: Enable strategic oscillation.
        feasibility_tolerance: Tolerance for infeasibility during oscillation.
        max_iterations: Maximum number of iterations.
        max_iterations_no_improve: Restart threshold.
        n_removal: Number of nodes to remove.
        n_llh: Number of low-level heuristics.
        time_limit: Wall-clock time limit.
        seed: Random seed.
        vrpp: Whether the problem is VRPP.
        profit_aware_operators: Whether to use profit-aware operators.
        use_swap: Enable swap neighborhood.
        use_relocate: Enable relocate neighborhood.
        use_2opt: Enable 2-opt neighborhood.
        use_insertion: Enable insertion-based neighborhoods.
        acceptance_criterion: Criterion for accepting moves.
    """

    # Short-term memory
    tabu_tenure: int = 7
    dynamic_tenure: bool = True
    min_tenure: int = 5
    max_tenure: int = 15

    # Aspiration criteria
    aspiration_enabled: bool = True

    # Long-term memory
    intensification_enabled: bool = True
    diversification_enabled: bool = True
    intensification_interval: int = 100
    diversification_interval: int = 200
    elite_size: int = 5
    frequency_penalty_weight: float = 0.1

    # Candidate list
    candidate_list_enabled: bool = True
    candidate_list_size: int = 20

    # Strategic oscillation
    oscillation_enabled: bool = False
    feasibility_tolerance: float = 0.1

    # General search
    max_iterations: int = 5000
    max_iterations_no_improve: int = 500
    n_removal: int = 3
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Neighborhood structure
    use_swap: bool = True
    use_relocate: bool = True
    use_2opt: bool = True
    use_insertion: bool = True

    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

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
    def from_config(cls, config: Any) -> TSParams:
        """Build parameters from a configuration source.

        Args:
            config: Configuration source.

        Returns:
            Instantiated TSParams.
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
            vrpp=getattr(config, "vrpp", True),
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

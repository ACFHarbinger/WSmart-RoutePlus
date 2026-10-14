"""Configuration parameters for the Iterated Local Search (ILS) solver.

Attributes:
    ILSParams: Parameter dataclass for the Iterated Local Search.

Example:
    >>> params = ILSParams(n_restarts=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class ILSParams:
    """Configuration for the ILS solver.

    Attributes:
        n_restarts: Number of perturbation + descent cycles.
        inner_iterations: LLH iterations per descent phase.
        n_removal: Nodes removed per LLH destroy step.
        n_llh: Number of LLHs in the pool.
        perturbation_strength: Fraction of nodes perturbed.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
        acceptance_criterion: Interface for solution acceptance.
    """

    n_restarts: int = 30
    inner_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    perturbation_strength: float = 0.15
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    def __post_init__(self):
        """Ensure acceptance criterion is initialized.

        Returns:
            None.
        """
        if self.acceptance_criterion is None:
            # Standard ILS uses Improving-Only acceptance
            from logic.src.policies.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

    @classmethod
    def from_config(cls, config: Any) -> ILSParams:
        """Build parameters from a configuration object.

        Args:
            config: Configuration source.

        Returns:
            Instantiated ILSParams.
        """
        # Build parameters
        params = cls(
            n_restarts=getattr(config, "n_restarts", 30),
            inner_iterations=getattr(config, "inner_iterations", 20),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            perturbation_strength=getattr(config, "perturbation_strength", 0.15),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
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
            # Default to only_improving for standard ILS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

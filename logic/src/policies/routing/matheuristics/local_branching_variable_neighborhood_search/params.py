"""
Configuration parameters for the Local Branching Variable Neighborhood Search (LB-VNS).

Attributes:
    LBVNSParams (LBVNSParams): Parameters for the LB-VNS policy.

Example:
    >>> from logic.src.policies.routing.matheuristics.local_branching_variable_neighborhood_search import LBVNSParams
    >>> params = LBVNSParams()
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory


@dataclass
class LBVNSParams:
    """
    Configuration parameters for LB-VNS.

    Attributes:
        k_min: Minimum neighborhood size.
        k_max: Maximum neighborhood size.
        k_step: Step size for neighborhood expansion.
        time_limit: Total time budget for optimization.
        time_limit_per_lb: Time limit for each Local Branching solve.
        max_lb_iterations: Maximum number of LB iterations per VNS iteration.
        mip_gap: Acceptable relative optimality gap.
        seed: Random seed.
    """

    k_min: int = 10
    k_max: int = 30
    k_step: int = 5
    time_limit: float = 300.0
    time_limit_per_lb: float = 60.0
    max_lb_iterations: int = 5
    mip_gap: float = 0.01
    seed: int = 42
    engine: str = "gurobi"
    framework: str = "ortools"

    acceptance_criterion: IAcceptanceCriterion = field(default_factory=lambda: None)  # type: ignore

    def __post_init__(self):
        """Ensure acceptance criterion is initialized even if not passed in config."""
        if self.acceptance_criterion is None:
            # Standard VNS uses Improving-Only acceptance
            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

    @classmethod
    def from_config(cls, config: Any) -> LBVNSParams:
        """Create LBVNSParams from a configuration object or dictionary.

        Args:
            config (Any): Configuration object.

        Returns:
            LBVNSParams: LBVNS parameters.
        """
        if isinstance(config, dict):
            params = cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})
        else:
            params = cls(
                k_min=getattr(config, "k_min", 10),
                k_max=getattr(config, "k_max", 30),
                k_step=getattr(config, "k_step", 5),
                time_limit=getattr(config, "time_limit", 300.0),
                time_limit_per_lb=getattr(config, "time_limit_per_lb", 60.0),
                max_lb_iterations=getattr(config, "max_lb_iterations", 5),
                mip_gap=getattr(config, "mip_gap", 0.01),
                seed=getattr(config, "seed", 42),
            )

        # Handle Acceptance Criterion Injection

        acceptance_cfg = getattr(config, "acceptance_criterion", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to only_improving for standard VNS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

    def to_dict(self) -> Dict[str, Any]:
        """Convert LBVNSParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of LBVNSParams.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

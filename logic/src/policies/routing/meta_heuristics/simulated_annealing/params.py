"""Configuration parameters for the Simulated Annealing (SA) meta-heuristic.

Attributes:
    SAParams: Parameter dataclass for the Simulated Annealing.

Example:
    >>> params = SAParams(initial_temperature=50.0)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class SAParams:
    """Hyper-parameters for the Simulated Annealing solver.

    Attributes:
        initial_temperature: Starting temperature T_0.
        cooling_rate: Geometric cooling coefficient alpha.
        min_temperature: Termination temperature threshold.
        target_acceptances_per_node: Target accepted moves per node.
        max_attempts_multiplier: Max attempts multiplier.
        frozen_streak_limit: Limit of iterations without improvement.
        auto_calibrate_temperature: Whether to auto-calibrate T_0.
        target_initial_acceptance: Target acceptance probability for calibration.
        calibration_samples: Samples used for calibration.
        n_restarts: Number of solver restarts.
        time_limit: Wall-clock time limit.
        seed: Random seed.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
        nb_granular: Number of granular neighbors.
        acceptance_criterion: Criterion for accepting moves.
    """

    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01
    target_acceptances_per_node: int = 10
    max_attempts_multiplier: int = 100
    frozen_streak_limit: int = 3
    auto_calibrate_temperature: bool = True
    target_initial_acceptance: float = 0.80
    calibration_samples: int = 200
    n_restarts: int = 1
    time_limit: float = 60.0
    seed: Optional[int] = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    nb_granular: int = 20
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: Any) -> SAParams:
        """Create SAParams from a configuration source.

        Args:
            config: Configuration source.

        Returns:
            Instantiated SAParams.
        """
        # Get base params
        if isinstance(config, dict):
            params = cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})
        else:
            params = cls(
                initial_temperature=getattr(config, "initial_temperature", 100.0),
                cooling_rate=getattr(config, "cooling_rate", 0.95),
                min_temperature=getattr(config, "min_temperature", 0.01),
                target_acceptances_per_node=getattr(config, "target_acceptances_per_node", 10),
                max_attempts_multiplier=getattr(config, "max_attempts_multiplier", 100),
                frozen_streak_limit=getattr(config, "frozen_streak_limit", 3),
                auto_calibrate_temperature=getattr(config, "auto_calibrate_temperature", True),
                target_initial_acceptance=getattr(config, "target_initial_acceptance", 0.80),
                calibration_samples=getattr(config, "calibration_samples", 200),
                n_restarts=getattr(config, "n_restarts", 1),
                time_limit=getattr(config, "time_limit", 60.0),
                seed=getattr(config, "seed", 42),
                vrpp=getattr(config, "vrpp", True),
                profit_aware_operators=getattr(config, "profit_aware_operators", False),
                nb_granular=getattr(config, "nb_granular", 20),
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
            # Legacy mapping
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.initial_temperature,
                alpha=params.cooling_rate,
                seed=params.seed,
            )

        return params

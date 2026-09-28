"""
Hyper-ACO Parameters Module.

This module defines the configuration parameters for the Hyper-Heuristic
Ant Colony Optimization algorithm. It uses a dataclass to store and validate
hyperparameters.

Attributes:
    None

Example:
    >>> from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import HyperACOParams
    >>> params = HyperACOParams(n_ants=10, alpha=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .hyper_operators import OPERATOR_NAMES


@dataclass
class HyperACOParams:
    """
    Parameters for Hyper-Heuristic ACO.

    Attributes:
        n_ants: Number of ants per iteration.
        alpha: Pheromone importance exponent.
        beta: Heuristic importance exponent.
        rho: Pheromone evaporation rate (ρ in the paper).
        eta_decay: Short-term visibility decay (ρ in the paper's visibility formula;
            kept separate from rho to allow independent tuning).
        tau_0: Initial pheromone level.
        Q: Pheromone floor constant added to every improving-journey deposit.
            Paper (Chen et al. 2007): Δτ^k_ij = Q + I_k / L_k when I_k > 0.
            Provides a minimum pheromone reward for any improving journey,
            preventing evaporation from erasing weak-but-consistent signals.
        lambda_val: Base for the visibility exponential when use_dynamic_lambda=False.
            Paper (Chen et al. 2007): λ = 1.0001, used as λ^{I_kj} so that negative
            improvements still yield a small positive visibility contribution.
        use_dynamic_lambda: If True (default), replace λ^I with exp(I / Z) where Z is
            a running EMA of |improvement|. This achieves scale invariance across
            problem sizes and is the recommended setting for the VRPP. Set to False for
            strict paper fidelity (λ = 1.0001 as base).
        max_iterations: Maximum number of iterations.
        elitism_ratio: Fraction of the swarm replaced by the global best when a new
            best solution is found. Paper (Chen et al. 2007) uses 1.0 (all ants sync).
            Lower values preserve population diversity at the cost of paper fidelity.
        time_limit: Maximum runtime in seconds.
        stagnation_limit: Number of iterations without improvement before pv is halved
            (Strategic Oscillation, per Chen et al. 2007).
        operators: Operator names the ants choose from; None (default) means all registered
            operators, which is what the archived runs used.
        sequence_length: Operators per journey; None (default) means one per operator, the
            journey length n of Chen et al. (2007) §III.B.
        time_weighted_visibility: If True, divide each visibility contribution by the operator's
            wall-clock runtime (a speed bonus). Off by default because it makes runs with the same
            seed differ; the count-normalised form is deterministic.
        vrpp: If True, operators consider unvisited nodes (VRPP mode).
        profit_aware_operators: If True, use profit-weighted removal/insertion operators.
        seed: Random seed for reproducibility.
    """

    n_ants: int = 10
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.5
    eta_decay: float = 0.5
    tau_0: float = 1.0
    Q: float = 1.0
    lambda_val: float = 1.0001
    use_dynamic_lambda: bool = True
    max_iterations: int = 50
    # Paper default is 1.0 (all ants sync); lower for diversity-preserving VRPP mode
    elitism_ratio: float = 1.0
    time_limit: float = 30.0
    stagnation_limit: int = 10
    operators: Optional[List[str]] = None
    sequence_length: Optional[int] = None
    time_weighted_visibility: bool = False
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters.

        Returns:
            None

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if self.Q < 0:
            raise ValueError(f"Pheromone floor constant Q must be >= 0, got {self.Q}")
        if not (0 < self.elitism_ratio <= 1):
            raise ValueError(f"elitism_ratio must be in (0, 1], got {self.elitism_ratio}")
        if self.operators is not None:
            unknown = [op for op in self.operators if op not in OPERATOR_NAMES]
            if unknown or not self.operators:
                raise ValueError(f"operators must be a non-empty subset of {OPERATOR_NAMES}; unknown: {unknown}")
        if self.sequence_length is not None and self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")

    @classmethod
    def from_config(cls, config: Any) -> HyperACOParams:
        """Create HyperACOParams from a configuration object.

        Args:
            config: The configuration object.

        Returns:
            HyperACOParams: The parameters for the HyperACO algorithm.
        """
        return cls(
            n_ants=getattr(config, "n_ants", 10),
            alpha=getattr(config, "alpha", 1.0),
            beta=getattr(config, "beta", 2.0),
            rho=getattr(config, "rho", 0.5),
            eta_decay=getattr(config, "eta_decay", 0.5),
            tau_0=getattr(config, "tau_0", 1.0),
            Q=getattr(config, "Q", 1.0),
            lambda_val=getattr(config, "lambda_val", 1.0001),
            use_dynamic_lambda=getattr(config, "use_dynamic_lambda", True),
            max_iterations=getattr(config, "max_iterations", 50),
            elitism_ratio=getattr(config, "elitism_ratio", 1.0),
            time_limit=getattr(config, "time_limit", 30.0),
            stagnation_limit=getattr(config, "stagnation_limit", 10),
            operators=getattr(config, "operators", None),
            sequence_length=getattr(config, "sequence_length", None),
            time_weighted_visibility=getattr(config, "time_weighted_visibility", False),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", None),
        )

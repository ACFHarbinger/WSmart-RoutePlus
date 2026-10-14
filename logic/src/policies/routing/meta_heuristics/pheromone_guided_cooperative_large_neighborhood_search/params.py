"""
Parameter dataclasses for the Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS).

This module defines all configuration structures used by PG-CLNS, providing
a single point of control for tuning its various components, including
Ant Colony Optimization (ACO) and Large Neighborhood Search (LNS).

Attributes:
    ACOParams: Parameters for Ant Colony Optimization.
    LNSParams: Configuration parameters for the LNS solver.
    PGCLNSParams: Top-level parameters; composes all subsystems.

Example:
    >>> params = PGCLNSParams(
    ...     population_size=10,
    ...     aco=ACOParams(n_ants=20),
    ...     lns=LNSParams(max_iterations=100)
    ... )
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# ACO Components (Initialization & Global Guidance)
# ---------------------------------------------------------------------------


@dataclass
class ACOParams:
    """
    Parameters for Ant Colony Optimization.

    Attributes:
        n_ants: Number of ants per iteration.
        k_sparse: Number of pheromone values to retain per node (candidate list size).
        alpha: Pheromone importance exponent.
        beta: Heuristic (distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        q0: Exploitation probability for pseudo-random proportional rule.
        tau_0: Initial pheromone level.
        tau_min: Minimum pheromone level (for MMAS bounds).
        tau_max: Maximum pheromone level (for MMAS bounds).
        max_iterations: Maximum number of iterations.
        time_limit: Maximum runtime in seconds.
        local_search: Whether to apply local search (2-opt) to solutions.
        local_search_iterations: Number of iterations for local search.
        elitist_weight: Weight for best-so-far solution in pheromone update.
    """

    n_ants: int = 20
    k_sparse: int = 15
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    q0: float = 0.9
    tau_0: Optional[float] = None
    tau_min: float = 0.001
    tau_max: float = 10.0
    max_iterations: int = 100
    time_limit: float = 30.0
    local_search: bool = True
    local_search_iterations: int = 100
    elitist_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if not (0 <= self.q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {self.q0}")


# ---------------------------------------------------------------------------
# LNS Components (Coaching & Improvement)
# ---------------------------------------------------------------------------


@dataclass
class LNSParams:
    """
    Configuration parameters for the LNS solver.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of LNS iterations.
        start_temp: Initial temperature for simulated annealing.
        cooling_rate: Temperature decay factor per iteration.
        reaction_factor: Learning rate for operator weight updates (rho).
        min_removal: Minimum number of nodes to remove.
        max_removal_pct: Maximum percentage of nodes to remove.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    max_removal_pct: float = 0.3


# ---------------------------------------------------------------------------
# Top-level HLNS Params (HVPL Population Dynamics)
# ---------------------------------------------------------------------------


@dataclass
class PGCLNSParams:
    """
    Parameters for the Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS) metaheuristic.

    Attributes:
        population_size: Population size.
        max_iterations: Number of iterations.
        replacement_rate: Fraction of population replaced.
        time_limit: Overall time limit in seconds.
        aco: ACO component parameters.
        lns: LNS component parameters.
    """

    # Population Parameters
    population_size: int = 10  # Population size
    max_iterations: int = 50  # Number of iterations
    replacement_rate: float = 0.2  # Fraction of population replaced
    time_limit: float = 60.0  # Overall time limit in seconds

    # ACO Components (Initialization & Global Guidance)
    aco: ACOParams = field(
        default_factory=lambda: ACOParams(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q0=0.9,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Only one iteration per construction phase
            time_limit=30,
            local_search=False,  # LNS handles local search
            local_search_iterations=0,
            elitist_weight=1.0,
        )
    )

    # LNS Components (Coaching & Improvement)
    lns: LNSParams = field(
        default_factory=lambda: LNSParams(
            max_iterations=100,  # "Coaching session" length
            start_temp=100.0,
            cooling_rate=0.95,
            reaction_factor=0.5,
            min_removal=1,
            max_removal_pct=0.2,
            time_limit=30,
        )
    )

    @classmethod
    def from_config(cls, config: Any) -> PGCLNSParams:
        """
        Build params from an OmegaConf / dataclass-like config object.

        Args:
            config: The configuration object.

        Returns:
            PGCLNSParams: The parameters as a PGCLNSParams object.
        """

        def _hydrate(sub_cls, sub_cfg: Any):
            kwargs: Dict[str, Any] = {}
            for f in fields(sub_cls):
                val = getattr(sub_cfg, f.name, MISSING)
                if val is MISSING or str(val) == "MISSING":
                    continue
                kwargs[f.name] = val
            return sub_cls(**kwargs)

        top_kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            sub_cfg = getattr(config, f.name, None)
            if sub_cfg is None:
                continue
            if f.name == "aco":
                top_kwargs[f.name] = _hydrate(ACOParams, sub_cfg)
            elif f.name == "lns":
                top_kwargs[f.name] = _hydrate(LNSParams, sub_cfg)
            else:
                val = getattr(config, f.name, MISSING)
                if val is MISSING or str(val) == "MISSING":
                    continue
                top_kwargs[f.name] = val
        return cls(**top_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the parameters to a dictionary.

        Returns:
            Dict[str, Any]: The parameters as a dictionary.
        """
        out: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "__dataclass_fields__"):
                out[f.name] = {g.name: getattr(val, g.name) for g in fields(val)}
            else:
                out[f.name] = val
        return out

"""
Configuration parameters for the Lin-Kernighan-Helsgaun 3 (LKH-3) policy.

Attributes:
    LKH3Params: Dataclass holding LKH-3 solver hyperparameters.

Example:
    >>> from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.params import LKH3Params
    >>> params = LKH3Params(runs=10, time_limit=60.0)
    >>> d = params.to_dict()
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Dict, List


@dataclass
class LKH3Params:
    """
    Configuration parameters for the LKH-3 solver.

    Attributes:
        runs: Number of independent runs (LKH precision).
        time_limit: Cumulative time limit across all runs.
        max_trials: Maximum trials per iteration (LKH trial limit).
        popmusic_subpath_size: Size of subpaths for POPMUSIC decomposition.
        popmusic_trials: Number of trials for each subpath.
        popmusic_max_candidates: Number of candidates per node in POPMUSIC.
        max_k_opt: Maximum number of swaps in the k-opt move (typically 5).
        use_ip_merging: Whether to use IP-based tour merging for final solution.
        max_pool_size: Size of the solution pool for tour merging.
        subgradient_iterations: Iterations for Lagrangian subgradient optimization.
        profit_aware_operators: Whether to use VRPP-specific operators.
        alns_iterations: Iterations for Adaptive Large Neighborhood Search (VRPP mode).
        plateau_limit: Iterations without improvement before triggering perturbation.
        deep_plateau_limit: Iterations for deeper search before restart.
        metaheuristic: Metaheuristic to use. Options: "ls_cgh" or "alns".
        sa_max_trials: Maximum number of trials for Simulated Annealing.
        perturb_operator_weights: Probability distribution for perturbation operators.
        seed: Random seed for reproducibility.
        vrpp: Whether to solve as a Vehicle Routing Problem with Profits.
    """

    runs: int = 10
    time_limit: float = 60.0
    max_trials: int = 1000
    popmusic_subpath_size: int = 50
    popmusic_trials: int = 50
    popmusic_max_candidates: int = 5
    max_k_opt: int = 5
    use_ip_merging: bool = True
    max_pool_size: int = 5
    subgradient_iterations: int = 50
    profit_aware_operators: bool = False
    alns_iterations: int = 100
    plateau_limit: int = 10
    deep_plateau_limit: int = 30
    sa_max_trials: int = 10000
    perturb_operator_weights: List[float] = dataclasses.field(default_factory=lambda: [0.6, 0.4])
    seed: int = 42
    vrpp: bool = True
    dynamic_topology_discovery: bool = False
    native_prize_collecting: bool = False

    @classmethod
    def from_config(cls, config: Any) -> LKH3Params:
        """Create LKH3Params from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with LKH-3 parameter attributes.

        Returns:
            LKH3Params: Populated parameter dataclass.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            runs=getattr(config, "runs", 10),
            time_limit=getattr(config, "time_limit", 60.0),
            max_trials=getattr(config, "max_trials", 1000),
            popmusic_subpath_size=getattr(config, "popmusic_subpath_size", 50),
            popmusic_trials=getattr(config, "popmusic_trials", 50),
            popmusic_max_candidates=getattr(config, "popmusic_max_candidates", 5),
            max_k_opt=getattr(config, "max_k_opt", 5),
            use_ip_merging=getattr(config, "use_ip_merging", True),
            max_pool_size=getattr(config, "max_pool_size", 5),
            subgradient_iterations=getattr(config, "subgradient_iterations", 50),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            alns_iterations=getattr(config, "alns_iterations", 100),
            plateau_limit=getattr(config, "plateau_limit", 10),
            deep_plateau_limit=getattr(config, "deep_plateau_limit", 30),
            sa_max_trials=getattr(config, "sa_max_trials", 10000),
            perturb_operator_weights=getattr(config, "perturb_operator_weights", [0.6, 0.4]),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            dynamic_topology_discovery=getattr(config, "dynamic_topology_discovery", False),
            native_prize_collecting=getattr(config, "native_prize_collecting", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

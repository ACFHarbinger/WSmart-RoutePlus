"""
Branch-and-Bound (BB) configuration schema.

This module defines the dataclass used to configure the Branch-and-Bound policy,
mapping YAML configuration values to typed Python attributes for use in the solver.

Attributes:
    BBConfig: Attributes for BB configuration.

Example:
    >>> from configs.policies.bb import BBConfig
    >>> config = BBConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class BBConfig:
    """Configuration for the Branch-and-Bound (BB) solver.

    The BB solver is an exact algorithm that explores a state-space search tree
    using Linear Programming (LP) relaxations to obtain lower bounds and prune
    branches that cannot yield a solution better than the current incumbent.

    **Formulation Selection**:
    The solver supports two mathematical formulations for subtour elimination:

    - **MTZ (Miller-Tucker-Zemlin)**: Compact formulation with O(n²) constraints.
      Uses load variables to prevent subtours. Custom B&B tree management with
      configurable branching strategies. Better for small-medium instances and
      when custom branching logic is desired.

    - **DFJ (Dantzig-Fulkerson-Johnson)**: Exponential constraints added lazily.
      Delegates B&B to Gurobi's internal engine. Better for large instances and
      when Gurobi license is available.

    Attributes:
        formulation (str): Mathematical formulation for subtour elimination.
            Supported: 'mtz', 'dfj', 'lr_uop'. Defaults to "dfj".

            - **"dfj"**: Dantzig-Fulkerson-Johnson with Gurobi's internal B&B engine
              and lazy subtour cuts. Best for large production instances.
            - **"mtz"**: Miller-Tucker-Zemlin compact formulation with a custom
              Python B&B tree. Best for small instances and ML-branching research.
            - **"lr_uop"**: Lagrangian Relaxation with an uncapacitated Orienteering
              Problem as the bounding oracle. Relaxes the capacity constraint with
              multiplier λ; tightens it via subgradient optimisation (Phase 1) before
              running a B&B tree (Phase 2). Best when capacity is the binding
              constraint and LP relaxations (MTZ/DFJ) are historically weak.

        time_limit (float): Maximum allowed runtime for the solver in seconds.
            Defaults to 60.0.
        mip_gap (float): Relative tolerance for the optimality gap. The solver
            terminates if (Incumbent - Bound) / Incumbent < mip_gap.
            Defaults to 0.01.
        branching_strategy (str): Logic for selecting the fractional variable to
            branch on. Only applies to MTZ formulation. Supported: 'most_fractional',
            'least_fractional', 'strong'. Defaults to "strong".
        strong_branching_limit (int): Maximum number of candidate variables to
            evaluate during strong branching. Only applies when branching_strategy='strong'.
            Defaults to 5.
            mode, where all bins are candidates for collection based on profitability.
            Defaults to True.
        seed (Optional[int]): Random seed passed to the underlying Gurobi engine
            for deterministic results. Defaults to None.
        packing (Optional[List[PackingConfig]]): List of
            strategies determining mandatory nodes to visit. Defaults to None.
        route_improvement (Optional[List[RouteImprovingConfig]]): List of operations
            to refine the tour after the exact search completes. Defaults to None.

        # LR-UOP only
        lr_lambda_init (float): Initial Lagrange multiplier λ₀ ≥ 0 for Phase 1.
            Defaults to 0.0.
        lr_max_subgradient_iters (int): Maximum Polyak subgradient iterations.
            Defaults to 100.
        lr_subgradient_theta (float): Step-size multiplier θ ∈ (0, 2] in the
            Polyak rule α_k = θ·(L(λ_k) - LB_k) / ‖g_k‖². Defaults to 1.0.
        lr_subgradient_time_fraction (float): Fraction of time_limit devoted to
            Phase 1 (subgradient). Remainder is used for Phase 2 (B&B).
            Defaults to 0.4.
        lr_op_time_limit (float): Per-call time limit for each uncapacitated OP
            solve during both phases (seconds). Defaults to 10.0.
        lr_branching_strategy (str): Customer selection rule for B&B branching.
            - "max_waste": branch on the free OP-selected customer with the
              highest waste load (most aggressively reduces capacity violation).
            - "min_profit": branch on the free customer with the smallest
              modified Lagrangian profit (R - λ*)·w_i.
            Defaults to "max_waste".
        lr_max_bb_nodes (int): Maximum B&B nodes to explore in Phase 2 before
            stopping (safety backstop). Defaults to 5000.
    """

    formulation: str = "dfj"
    time_limit: float = 60.0
    mip_gap: float = 0.01
    branching_strategy: str = "strong"
    strong_branching_limit: int = 5
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    # LR-UOP: Subgradient phase
    lr_lambda_init: float = 0.0
    lr_max_subgradient_iters: int = 100
    lr_subgradient_theta: float = 1.0
    lr_subgradient_time_fraction: float = 0.4

    # LR-UOP: Inner OP solver
    lr_op_time_limit: float = 10.0

    # LR-UOP: B&B phase
    lr_branching_strategy: str = "max_waste"
    lr_max_bb_nodes: int = 5000

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BBConfig":
        """Create a BBConfig instance from a raw configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A BBConfig instance with values mapped from the config.
        """
        return cls(
            time_limit=float(config.get("time_limit", 60.0)),
            mip_gap=float(config.get("mip_gap", 0.01)),
            seed=int(config.get("seed", 42)),
            branching_strategy=str(config.get("branching_strategy", "strong")),
            strong_branching_limit=int(config.get("strong_branching_limit", 5)),
            formulation=str(config.get("formulation", "dfj")),
            # LR-UOP parameters
            lr_lambda_init=float(config.get("lr_lambda_init", 0.0)),
            lr_max_subgradient_iters=int(config.get("lr_max_subgradient_iters", 100)),
            lr_subgradient_theta=float(config.get("lr_subgradient_theta", 1.0)),
            lr_subgradient_time_fraction=float(config.get("lr_subgradient_time_fraction", 0.4)),
            lr_op_time_limit=float(config.get("lr_op_time_limit", 10.0)),
            lr_branching_strategy=str(config.get("lr_branching_strategy", "max_waste")),
            lr_max_bb_nodes=int(config.get("lr_max_bb_nodes", 5000)),
        )

"""
Configuration for Local Branching with Variable Neighborhood Search (LB-VNS).

Attributes:
    LocalBranchingVNSConfig: Configuration for the Local Branching with Variable Neighborhood Search (LB-VNS) policy.

Example:
    >>> from configs.policies.lb_vns import LocalBranchingVNSConfig
    >>> config = LocalBranchingVNSConfig()
    >>> config.time_limit
    300.0
    >>> config.k_min
    10
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig
from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class LocalBranchingVNSConfig:
    """
    Configuration for the Local Branching with Variable Neighborhood Search (LB-VNS) matheuristic.

    LB-VNS (Hansen, Mladenović, and Urošević, 2006) is a powerful hybrid that
    integrates the metaheuristic structure of Variable Neighborhood Search (VNS)
    with the tactical 'local' search capabilities of Local Branching (LB).

    The algorithm operates by systematically exploring neighborhoods of increasing
    size (defined by the Hamming distance 'k') whenever the current neighborhood
    fails to yield an improved solution. This allows the solver to escape local
    optima while maintaining the mathematical rigor of a MILP-based intensification.

    Attributes:
        time_limit (float): Total wall-clock time limit for the entire VNS process.
            If exceeded, the algorithm terminates and returns the best found solution.

        k_min (int): Initial neighborhood size (Hamming distance). This determines
             the 'radius' of the first restricted search space. Standard values range from 10 to 20.

        k_max (int): Maximum neighborhood size for the VNS loop. If LB-VNS reaches this
            size without finding an improvement, it may reset or terminate depending on
            the implementation. It acts as an upper bound on diversification intensity.

        k_step (int): Increment value for neighborhood expansion. Whenever a sub-problem
            is solved to optimality (or a limit) without finding a better solution,
            k is increased by this amount: k = k + k_step.

        time_limit_per_lb (float): Maximum time budget allocated for each internal
            Local Branching execution (local search phase). This ensures the solver
            doesn't get stuck too long in a single, potentially difficult, sub-MIP.

        max_lb_iterations (int): Limit on the number of internal improvement steps
            taken within each Local Branching call. In LB-VNS, the LB part itself
            can iterate to find a local optimum before the VNS moves to the next neighborhood.

        mip_gap (float): Relative optimality gap targeted for each sub-problem solve.
            A value like 0.01 (1%) allows the solver to stop once it's close enough to
            the local optimal solution, saving time for more VNS explorations.

        seed (int): Global random seed for both the Shaking mechanism (random perturbation)
            and the underlying Gurobi solver, ensuring deterministic and reproducible results.


        # Infrastructure Settings
        # -----------------------
        engine (str): Solver engine to use ('gurobi', 'scip', 'highs', or 'cplex').

        framework (str): Solver framework to use ('ortools', 'pyomo').

        packing (Optional[PackingConfig]): Configuration for identifying
            and enforcing mandatory nodes in the routing problem.

        route_improvement (Optional[RouteImprovingConfig]): Optional local search or
            heuristics (e.g., node swaps, 2-opt) to be applied to solutions found
            by the matheuristic for further minor refinement.

        acceptance_criterion (AcceptanceConfig): Acceptance criterion config for local search.
    """

    time_limit: float = 300.0
    k_min: int = 10
    k_max: int = 60
    k_step: int = 10
    time_limit_per_lb: float = 30.0
    max_lb_iterations: int = 15
    mip_gap: float = 0.01
    seed: int = 42

    # Infrastructure
    engine: str = "gurobi"
    framework: str = "ortools"
    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="only_improving"))

    def __post_init__(self):
        """Ensure acceptance criterion is initialized even if not passed in config."""
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        if self.acceptance_criterion is None:
            # Standard VNS uses Improving-Only acceptance
            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

    @classmethod
    def from_config(cls, config: Any) -> "LocalBranchingVNSConfig":
        """Create LocalBranchingVNSConfig from a configuration object or dictionary.

        Args:
            config (Any): Configuration object.

        Returns:
            "LocalBranchingVNSConfig": LBVNS parameters.
        """
        from dataclasses import fields

        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

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
        """Convert LocalBranchingVNSConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of LocalBranchingVNSConfig.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}

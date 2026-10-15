r"""
Parameter dataclasses for the Hybrid Memetic Large Neighborhood Search (HMLNS).

This module defines all configuration structures used by HMLNS, providing
a single point of control for tuning and ablating its various components.

Attributes:
    MACOParams: Parameters for Memetic Ant Colony Optimization.
    HybridMemeticLargeNeighborhoodSearchParams: Top-level parameters composing all subsystems.

Example:
    >>> params = HybridMemeticLargeNeighborhoodSearchParams(
    ...     population_size=50,
    ...     aco_params=MACOParams(n_ants=30),
    ...     alns_params=ALNSConfig(max_iterations=1000)
    ... )
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from logic.src.configs.policies.alns import ALNSConfig

if TYPE_CHECKING:
    from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

# ---------------------------------------------------------------------------
# MACO (Memetic Ant Colony Optimization)
# ---------------------------------------------------------------------------


@dataclass
class MACOParams:
    """
    Parameters for Memetic Ant Colony Optimization (K-Sparse MMAS variant).

    Canonical MMAS (Stützle & Hoos 2000) parameters are documented below
    alongside the memetic extensions introduced by this implementation.

    -- Core ACO parameters --

    Attributes:
        n_ants: Number of ants per iteration.
        k_sparse: Size of candidate lists (k-nearest neighbors for each node).
        alpha: Pheromone importance exponent.
        beta: Heuristic (distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        scale: Precision parameter for pheromone pruning (Hale 2021).

    -- MMAS pheromone bound parameters --

        tau_0: Initial pheromone level (None → 1 / (n * C_nn)).
        tau_min: Minimum pheromone bound. If None, computed dynamically from
            tau_max and p_best: tau_min = tau_max * (1 − p_best^(1/n))^(avg/2).
            Set to a positive float to use a fixed floor instead.
        tau_max: Maximum pheromone bound. If None, computed dynamically from
            the best solution cost: tau_max = 1 / (rho * C_bs). Set to a
            positive float to use a fixed ceiling instead.
        p_best: Probability that the best solution is constructed at convergence,
            used for the dynamic tau_min formula (Stützle & Hoos 2000 recommend
            0.05). Only used when tau_min is None.

    -- MMAS update schedule --

        update_schedule: Controls which solution reinforces pheromones each
            iteration. Options:
            - "best_so_far":  always use the global best (bs) solution.
            - "iteration_best": always use the current iteration's best (ib).
            - "auto":         use iteration_best for the first
                              ``ib_phase_length`` iterations, then switch to
                              best_so_far (canonical MMAS schedule).
        ib_phase_length: Number of iterations to use iteration_best before
            switching to best_so_far. Only used when update_schedule="auto".
            Defaults to 25% of max_iterations if set to 0.
        elitist_weight: Weight w ∈ [0, 1] for the elitist (best-so-far)
            solution in a blended update:
                delta = w * delta_bs + (1 − w) * delta_ib
            When update_schedule != "auto", the non-selected solution receives
            (1 − elitist_weight) weight. Set to 1.0 to use a single solution.

    -- Stagnation and restart --

        stagnation_limit: Number of iterations without improvement before a
            pheromone reinitialization is triggered. Set to 0 to disable.
        restart_pheromone: Pheromone level to reset to on restart. If None,
            resets to the current tau_max.
        use_restart_best: If True, pheromone reinforcement after a restart
            uses the best-so-far solution (not the iteration best). This
            preserves exploration while re-biasing toward the known optimum.

    -- Memetic / local search --

        local_search: Whether to apply local search to each ant's solution.
        local_search_iterations: Number of inner iterations for local search.
        elite_pool_size: Number of elite solutions to maintain in memory.
            These are used for population-level crossover in the memetic layer.
            Set to 0 to disable the elite pool entirely.

    -- Miscellaneous --

        max_iterations: Maximum number of iterations.
        time_limit: Maximum runtime in seconds.
        profit_aware_operators: Whether to use profit-aware operators.
        seed: Random seed.
        acceptance_criterion: Acceptance criterion for solutions.
    """

    # -- Core ACO --
    n_ants: int = 20
    k_sparse: int = 15
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    scale: float = 5.0

    # -- Pheromone bounds --
    tau_0: Optional[float] = None
    tau_min: Optional[float] = None  # None → dynamic (p_best formula)
    tau_max: Optional[float] = None  # None → dynamic (1 / rho*C_bs)
    p_best: float = 0.05  # used when tau_min is None

    # -- Update schedule --
    update_schedule: str = "auto"  # "auto" | "best_so_far" | "iteration_best"
    ib_phase_length: int = 0  # 0 → 25% of max_iterations
    elitist_weight: float = 1.0  # 1.0 → single-solution update

    # -- Stagnation / restart --
    stagnation_limit: int = 25  # 0 → disabled
    restart_pheromone: Optional[float] = None
    use_restart_best: bool = True

    # -- Memetic / LS --
    local_search: bool = False
    local_search_iterations: int = 100
    elite_pool_size: int = 5  # 0 → disabled

    # -- Termination --
    max_iterations: int = 100
    time_limit: float = 30.0

    # -- Problem / misc --
    profit_aware_operators: bool = False
    seed: Optional[int] = 42
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    def __post_init__(self) -> None:
        """Validate parameters.

        Returns:
            None

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if self.scale < 0:
            raise ValueError(f"Scale parameter must be non-negative, got {self.scale}")
        valid_schedules = {"auto", "best_so_far", "iteration_best"}
        if self.update_schedule not in valid_schedules:
            raise ValueError(f"update_schedule must be one of {valid_schedules}, got '{self.update_schedule}'")
        if not (0.0 < self.p_best < 1.0):
            raise ValueError(f"p_best must be in (0, 1), got {self.p_best}")
        if not (0.0 <= self.elitist_weight <= 1.0):
            raise ValueError(f"elitist_weight must be in [0, 1], got {self.elitist_weight}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MACOParams:
        """Create MACOParams from a configuration object.

        Args:
            config: The configuration object.

        Returns:
            MACOParams: The parameters for the MACO algorithm.
        """
        if config is None:
            return cls()

        raw_data: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw_data = config
        else:
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw_data[f.name] = getattr(config, f.name)

        kwargs: Dict[str, Any] = {}

        for f in fields(cls):
            if f.name == "acceptance_criterion":
                continue
            val = raw_data.get(f.name, getattr(cls, f.name, f.default))
            if val is not None:
                if f.type is float or f.type == "float":
                    val = float(val)
                elif f.type is int or f.type == "int":
                    val = int(val)
                elif f.type is bool or f.type == "bool":
                    val = val.lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)
            kwargs[f.name] = val

        params = cls(**kwargs)

        # Handle Acceptance Criterion Injection
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = (
            raw_data.get("acceptance_criterion")
            if isinstance(config, dict)
            else getattr(config, "acceptance_criterion", None)
        )
        if acceptance_cfg:
            if hasattr(acceptance_cfg, "method"):
                name = acceptance_cfg.method
                params_cfg = getattr(acceptance_cfg, "params", {})
            else:
                name = acceptance_cfg.get("method", "oi")
                params_cfg = acceptance_cfg.get("params", {})

            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=name,
                config=params_cfg,
            )
        else:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params


# ---------------------------------------------------------------------------
# Top-level HMLNS Params
# ---------------------------------------------------------------------------


@dataclass
class HybridMemeticLargeNeighborhoodSearchParams:
    """
    Configuration parameters for Hybrid Memetic Large Neighborhood Search (HMLNS).

    Composes ACO and ALNS sub-parameters into a unified evolutionary framework.

    Attributes:
        population_size: Number of chromosomes in the active population.
        max_generations: Total number of evolutionary generations.
        substitution_rate: Fraction of population replaced by reserve pool.
        crossover_rate: Probability of recombining two parents.
        mutation_rate: Probability of local perturbation.
        elitism_count: Number of best solutions to preserve across generations.
        aco_init_iterations: ACO iterations for population seeding.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        profit_aware_operators: Whether to use profit-aware heuristics.
        aco_params: MACO configuration.
        alns_params: ALNS configuration.
    """

    # Population Parameters
    population_size: int = 30
    max_generations: int = 100
    substitution_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 3

    # Phase-specific Parameters
    n_removal: int = 3
    aco_init_iterations: int = 50
    time_limit: float = 300.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False

    # Sub-algorithm Parameters
    aco_params: MACOParams = field(default_factory=MACOParams)
    alns_params: ALNSConfig = field(default_factory=ALNSConfig)

    def __post_init__(self) -> None:
        """Forward global flags to sub-parameters."""
        self.aco_params.profit_aware_operators = self.profit_aware_operators
        self.aco_params.seed = self.seed

        self.alns_params.profit_aware_operators = self.profit_aware_operators
        self.alns_params.seed = self.seed

    @classmethod
    def from_config(cls, config: Any) -> HybridMemeticLargeNeighborhoodSearchParams:
        """
        Build params from an OmegaConf / dataclass-like config object.

        Follows the CALM pattern for hydrating nested subsystems.

        Args:
            config: The configuration object.

        Returns:
            HybridMemeticLargeNeighborhoodSearchParams: The parameters for the HMLNS algorithm.
        """

        def _hydrate(sub_cls: Type, sub_cfg: Any) -> Any:
            if isinstance(sub_cfg, sub_cls):
                return sub_cfg
            if hasattr(sub_cls, "from_config"):
                return sub_cls.from_config(sub_cfg)
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
                # Top-level fields
                val = getattr(config, f.name, MISSING)
                if val is not MISSING and str(val) != "MISSING":
                    top_kwargs[f.name] = val
                continue

            if f.name == "aco_params":
                top_kwargs[f.name] = _hydrate(MACOParams, sub_cfg)
            elif f.name == "alns_params":
                top_kwargs[f.name] = _hydrate(ALNSConfig, sub_cfg)
            else:
                top_kwargs[f.name] = sub_cfg

        return cls(**top_kwargs)

    @property
    def max_iterations(self) -> int:
        """Alias for max_generations for legacy compatibility.

        Returns:
            int: The maximum number of iterations.
        """
        return self.max_generations

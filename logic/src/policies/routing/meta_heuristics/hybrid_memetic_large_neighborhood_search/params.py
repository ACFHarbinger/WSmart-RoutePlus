r"""
Parameter dataclasses for the Hybrid Memetic Large Neighborhood Search (HMLNS).

This module defines all configuration structures used by HMLNS, providing
a single point of control for tuning and ablating its various components.

Attributes:
    MACOParams: Parameters for Memetic Ant Colony Optimization.
    ALNSParams: Parameters for Adaptive Large Neighborhood Search.
    HybridMemeticLargeNeighborhoodSearchParams: Top-level parameters composing all subsystems.

Example:
    >>> params = HybridMemeticLargeNeighborhoodSearchParams(
    ...     population_size=50,
    ...     aco_params=MACOParams(n_ants=30),
    ...     alns_params=ALNSParams(max_iterations=1000)
    ... )
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

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
        vrpp: Whether to use VRPP mode.
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
    vrpp: bool = True
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
# ALNS (Adaptive Large Neighborhood Search)
# ---------------------------------------------------------------------------


@dataclass
class ALNSParams:
    """
    Configuration parameters for the ALNS solver.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing (if > 0, dynamic calculation is disabled).
        cooling_rate: Temperature decay factor per iteration.
        reaction_factor: Learning rate for operator weight updates (r in Ropke & Pisinger 2005).
        min_removal: Minimum number of nodes to remove.
        max_removal_pct: Maximum percentage of nodes to remove.
        segment_size: Number of iterations before updating operator weights.
        noise_factor: Noise level for repair operators (eta in Ropke & Pisinger 2005).
        worst_removal_randomness: Randomness parameter p >= 1 for worst removal (p=1 is deterministic).
        start_temp_control: Controls initial temperature (0.05 = accept 5% worse with 0.5 probability).
        xi: Fraction of total nodes for max removal cap.
        sigma_1: Score for new global best solution.
        sigma_2: Score for better solution (not global best, not visited before).
        sigma_3: Score for accepted worse solution (not visited before).
        vrpp: If True, allow expanding insertion pool beyond removed nodes.
        profit_aware_operators: If True, use profit-aware insertion/removal operators.
        extended_operators: If True, add string, cluster, and neighbor removal operators.
        seed: Random seed for reproducibility.
        engine: Engine to use for the solver.
        acceptance_criterion: Acceptance criterion for solutions.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 0.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 4
    max_removal_pct: float = 0.3
    segment_size: int = 100
    noise_factor: float = 0.025
    worst_removal_randomness: float = 3.0
    shaw_randomization: float = 6.0
    max_removal_cap: int = 100
    start_temp_control: float = 0.05  # 'w' parameter: accept a 5% worse solution with 0.5 probability
    xi: float = 0.4  # 'xi' parameter: fraction of total nodes for max removal cap
    regret_pool: str = "regret234"
    sigma_1: float = 33.0
    sigma_2: float = 9.0
    sigma_3: float = 13.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    extended_operators: bool = False
    seed: Optional[int] = None
    engine: str = "custom"
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: Any) -> ALNSParams:
        """Create ALNSParams from an ALNSConfig dataclass or dict.

        Args:
            config: The configuration object.

        Returns:
            ALNSParams: The parameters for the ALNS algorithm.
        """
        if config is None:
            return cls()

        if isinstance(config, dict):
            params = cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})
        else:
            params = cls(
                time_limit=getattr(config, "time_limit", 60.0),
                max_iterations=getattr(config, "max_iterations", 5000),
                start_temp=getattr(config, "start_temp", 0.0),
                cooling_rate=getattr(config, "cooling_rate", 0.995),
                reaction_factor=getattr(config, "reaction_factor", 0.1),
                min_removal=getattr(config, "min_removal", 4),
                max_removal_pct=getattr(config, "max_removal_pct", 0.3),
                segment_size=getattr(config, "segment_size", 100),
                noise_factor=getattr(config, "noise_factor", 0.025),
                worst_removal_randomness=getattr(config, "worst_removal_randomness", 3.0),
                shaw_randomization=getattr(config, "shaw_randomization", 6.0),
                max_removal_cap=getattr(config, "max_removal_cap", 100),
                regret_pool=getattr(config, "regret_pool", "regret234"),
                sigma_1=getattr(config, "sigma_1", 33.0),
                sigma_2=getattr(config, "sigma_2", 9.0),
                sigma_3=getattr(config, "sigma_3", 13.0),
                vrpp=getattr(config, "vrpp", True),
                profit_aware_operators=getattr(config, "profit_aware_operators", False),
                extended_operators=getattr(config, "extended_operators", False),
                seed=getattr(config, "seed", None),
                engine=getattr(config, "engine", "custom"),
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
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.start_temp,
                alpha=params.cooling_rate,
                seed=params.seed,
            )

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
        vrpp: Whether to solve the VRP with profits.
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
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Sub-algorithm Parameters
    aco_params: MACOParams = field(default_factory=MACOParams)
    alns_params: ALNSParams = field(default_factory=ALNSParams)

    def __post_init__(self) -> None:
        """Forward global flags to sub-parameters."""
        self.aco_params.vrpp = self.vrpp
        self.aco_params.profit_aware_operators = self.profit_aware_operators
        self.aco_params.seed = self.seed

        self.alns_params.vrpp = self.vrpp
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
                top_kwargs[f.name] = _hydrate(ALNSParams, sub_cfg)
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
